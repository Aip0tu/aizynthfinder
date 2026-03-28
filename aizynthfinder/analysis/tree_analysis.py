"""用于分析树搜索结果的模块。"""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from paretoset import paretorank, paretoset

from aizynthfinder.analysis.utils import RouteSelectionArguments
from aizynthfinder.chem import FixedRetroReaction, hash_reactions
from aizynthfinder.context.scoring import StateScorer
from aizynthfinder.reactiontree import ReactionTree
from aizynthfinder.search.andor_trees import AndOrSearchTreeBase
from aizynthfinder.search.mcts import MctsNode, MctsSearchTree

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.context.scoring import Scorer
    from aizynthfinder.utils.type_utils import (
        Any,
        Dict,
        Iterable,
        List,
        Optional,
        Sequence,
        StrDict,
        Tuple,
        Union,
    )

    _Solution = Union[MctsNode, ReactionTree]
    _AnyListOfSolutions = Union[Sequence[MctsNode], Sequence[ReactionTree]]
    _AnyListOfReactions = Sequence[
        Union[Iterable[RetroReaction], Iterable[FixedRetroReaction]]
    ]


class TreeAnalysis:
    """
    封装对搜索树进行的多种分析操作。

    :ivar scorers: 用于给节点打分的评分器对象
    :ivar search_tree: 被分析的搜索树

    :param search_tree: 待分析的搜索树
    :param scorer: 用于给节点打分的对象，默认为 `StateScorer`
    """

    def __init__(
        self,
        search_tree: Union[MctsSearchTree, AndOrSearchTreeBase],
        scorer: Optional[Union[Scorer, List[Scorer]]] = None,
    ) -> None:
        self.search_tree = search_tree
        scorer = scorer or StateScorer(search_tree.config)
        if isinstance(scorer, list):
            self.scorers: List[Scorer] = scorer
        else:
            self.scorers = [scorer]
        self._direction = "max"  # 当前实现默认所有目标都需要最大化。
        self._single_objective = len(self.scorers) == 1

    def best(self) -> _Solution:
        """
        返回得分最高的路线或 MCTS 风格节点。

        如果多条路线分数相同，则返回第一条。

        :return: 得分最高的节点或路线
        :raises ValueError: 当当前分析是多目标分析时抛出
        """
        if not self._single_objective:
            raise ValueError("Cannot return best item for multi-objective analysis")

        if isinstance(self.search_tree, MctsSearchTree):
            nodes = self._all_nodes()
            sorted_nodes, _, _ = self.scorers[0].sort(nodes)
            return sorted_nodes[0]

        sorted_routes, _, _ = self.scorers[0].sort(self.search_tree.routes())
        return sorted_routes[0]

    def pareto_front(self) -> Tuple[_Solution, ...]:
        """
        返回多目标搜索中的帕累托前沿路线或 MCTS 风格节点。

        :returns: 帕累托前沿上的解
        :raises ValueError: 当当前分析是单目标分析时抛出
        """
        if self._single_objective:
            raise ValueError("Cannot return Pareto front for single-objective analysis")

        if isinstance(self.search_tree, MctsSearchTree):
            solutions = self.search_tree.nodes()
        else:
            solutions = self.search_tree.routes()  # type: ignore

        scores_arr = np.array(
            [[scorer(solution) for scorer in self.scorers] for solution in solutions]
        )
        direction_arr = np.repeat(self._direction, len(self.scorers))
        pareto_mask = paretoset(scores_arr, sense=direction_arr, distinct=False)
        pareto_idxs = np.arange(len(solutions))[pareto_mask]
        pareto_front: Sequence[_Solution] = [solutions[idx] for idx in pareto_idxs]

        return tuple(pareto_front)

    def sort(
        self, selection: Optional[RouteSelectionArguments] = None
    ) -> Tuple[_AnyListOfSolutions, Sequence[Dict[str, float]]]:
        """
        对搜索树中的节点或路线进行排序和筛选。

        每个解的分数都会以字典形式返回，
        其中每个目标对应一个分数字段。

        :param selection: 路线筛选条件
        :return: 排序并筛选后的条目
        :return: 对应条目的分数字典
        """
        selection = selection or RouteSelectionArguments()

        if not self._single_objective:
            return self._pareto_rank_sort(selection)

        scorer = self.scorers[0]
        if isinstance(self.search_tree, MctsSearchTree):
            nodes = self._all_nodes()
            sorted_items, sorted_scores, _ = scorer.sort(nodes)
            actions = [node.actions_to() for node in sorted_items]

        else:
            sorted_items, sorted_scores, _ = scorer.sort(self.search_tree.routes())
            actions = [route.reactions() for route in sorted_items]

        scores = [{repr(scorer): score} for score in sorted_scores]

        return self._collect_top_items(
            sorted_items, scores, actions, sorted_scores, selection, True
        )

    def tree_statistics(self) -> StrDict:
        """
        返回搜索树的统计信息。

        当前会返回节点数、最大变换次数、最大子节点数、最高得分、
        最高分路线是否已求解、最高分节点中的分子数量，以及前体相关信息。

        :return: 统计信息字典
        """
        if isinstance(self.search_tree, MctsSearchTree):
            return self._tree_statistics_mcts()
        return self._tree_statistics_andor()

    def _all_nodes(self) -> Sequence[MctsNode]:
        assert isinstance(self.search_tree, MctsSearchTree)
        # 这里是为了保持向后兼容，后续仍值得进一步梳理。
        if repr(self.scorers[0]) == "state score":
            return list(self.search_tree.graph())
        return [node for node in self.search_tree.graph() if not node.children]

    def _pareto_rank_sort(
        self,
        selection: RouteSelectionArguments,
    ) -> Tuple[_AnyListOfSolutions, Sequence[Dict[str, float]]]:
        if isinstance(self.search_tree, MctsSearchTree):
            solutions = self._all_nodes()
        else:
            solutions = self.search_tree.routes()  # type: ignore

        scores_arr = np.array(
            [[scorer(solution) for scorer in self.scorers] for solution in solutions]
        )
        direction_arr = np.repeat(self._direction, len(self.scorers))
        pareto_ranks = paretorank(scores_arr, sense=direction_arr, distinct=False)

        sortidx = sorted(range(len(pareto_ranks)), key=pareto_ranks.__getitem__)
        sorted_pareto_ranks = sorted(pareto_ranks)
        sorted_scores = [
            {
                repr(scorer): scores_arr[idx, scorer_idx]
                for scorer_idx, scorer in enumerate(self.scorers)
            }
            for idx in sortidx
        ]
        sorted_items = [solutions[idx] for idx in sortidx]

        if isinstance(self.search_tree, MctsSearchTree):
            actions = [node.actions_to() for node in sorted_items]
        else:
            actions = [route.reactions() for route in sorted_items]  # type: ignore

        return self._collect_top_items(
            sorted_items, sorted_scores, actions, sorted_pareto_ranks, selection, False
        )

    def _top_nodes(self) -> Tuple[_Solution, ...]:
        if self._single_objective:
            return (self.best(),)
        return self.pareto_front()

    def _tree_statistics_andor(self) -> StrDict:
        assert isinstance(self.search_tree, AndOrSearchTreeBase)
        top_routes = self._top_nodes()
        mols_in_stock = self._top_ranked_join(
            ", ".join(mol.smiles for mol in route.leafs() if route.in_stock(mol))  # type: ignore
            for route in top_routes
        )
        mols_not_in_stock = self._top_ranked_join(
            ", ".join(mol.smiles for mol in route.leafs() if not route.in_stock(mol))  # type: ignore
            for route in top_routes
        )
        all_routes = self.search_tree.routes()
        policy_used_counts = self._policy_used_statistics(
            [reaction for route in all_routes for reaction in route.reactions()]
        )
        availability = self._top_ranked_join(
            ";".join(
                self.search_tree.config.stock.availability_string(mol)
                for mol in route.leafs()  # type: ignore
            )
            for route in top_routes
        )
        number_of_precursors_in_stock = self._top_ranked_join(
            sum(route.in_stock(leaf) for leaf in route.leafs()) for route in top_routes  # type: ignore
        )

        # 多目标情况下若把全部分数都塞进这里会显得过于杂乱，
        # 这些分数仍会随路线结果一起返回。
        if self._single_objective:
            top_score = self.scorers[0](top_routes[0])
        else:
            top_score = None

        return {
            "number_of_nodes": len(self.search_tree.mol_nodes),
            "max_transforms": max(
                node.prop["mol"].transform for node in self.search_tree.mol_nodes
            ),
            "max_children": max(
                len(node.children) for node in self.search_tree.mol_nodes
            ),
            "number_of_routes": len(all_routes),
            "number_of_solved_routes": sum(route.is_solved for route in all_routes),
            "top_score": top_score,
            "is_solved": self._top_ranked_join(route.is_solved for route in top_routes),
            "number_of_steps": self._top_ranked_join(
                len(list(route.reactions())) for route in top_routes  # type: ignore
            ),
            "number_of_precursors": self._top_ranked_join(
                len(list(route.leafs())) for route in top_routes  # type: ignore
            ),
            "number_of_precursors_in_stock": number_of_precursors_in_stock,
            "precursors_in_stock": mols_in_stock,
            "precursors_not_in_stock": mols_not_in_stock,
            "precursors_availability": availability,
            "policy_used_counts": policy_used_counts,
            "profiling": getattr(self.search_tree, "profiling", {}),
        }

    def _tree_statistics_mcts(self) -> StrDict:
        assert isinstance(self.search_tree, MctsSearchTree)
        top_nodes = self._top_nodes()
        assert isinstance(top_nodes[0], MctsNode)
        top_states = [node.state for node in top_nodes]  # type: ignore
        nodes = list(self.search_tree.graph())
        mols_in_stock = self._top_ranked_join(
            ", ".join(
                mol.smiles
                for mol, instock in zip(state.mols, state.in_stock_list)
                if instock
            )
            for state in top_states
        )
        mols_not_in_stock = self._top_ranked_join(
            ", ".join(
                mol.smiles
                for mol, instock in zip(state.mols, state.in_stock_list)
                if not instock
            )
            for state in top_states
        )

        policy_used_counts = self._policy_used_statistics(
            [node[child]["action"] for node in nodes for child in node.children]
        )

        # 多目标情况下若把全部分数都塞进这里会显得过于杂乱，
        # 这些分数仍会随路线结果一起返回。
        if self._single_objective:
            top_score = self.scorers[0](top_nodes[0])
        else:
            top_score = None

        return {
            "number_of_nodes": len(nodes),
            "max_transforms": max(node.state.max_transforms for node in nodes),
            "max_children": max(len(node.children) for node in nodes),
            "number_of_routes": sum(1 for node in nodes if not node.children),
            "number_of_solved_routes": sum(
                1 for node in nodes if not node.children and node.state.is_solved
            ),
            "top_score": top_score,
            "is_solved": self._top_ranked_join(state.is_solved for state in top_states),
            "number_of_steps": self._top_ranked_join(
                state.max_transforms for state in top_states
            ),
            "number_of_precursors": self._top_ranked_join(
                len(state.mols) for state in top_states
            ),
            "number_of_precursors_in_stock": self._top_ranked_join(
                sum(state.in_stock_list) for state in top_states
            ),
            "precursors_in_stock": mols_in_stock,
            "precursors_not_in_stock": mols_not_in_stock,
            "precursors_availability": self._top_ranked_join(
                ";".join(state.stock_availability) for state in top_states
            ),
            "policy_used_counts": policy_used_counts,
            "profiling": getattr(self.search_tree, "profiling", {}),
        }

    @staticmethod
    def _collect_top_items(
        items: _AnyListOfSolutions,
        scores: Sequence[Dict[str, float]],
        reactions: _AnyListOfReactions,
        ranks: Sequence[Union[int, float]],
        selection: RouteSelectionArguments,
        min_comp: bool,
    ) -> Tuple[_AnyListOfSolutions, Sequence[Dict[str, float]]]:
        """
        找出并返回排名最高的超节点或反应树及其分数。

        `selection` 参数用于控制返回条目数量。
        如果至少有一个条目已求解，并且 `selection` 指定返回全部，
        则会返回所有已求解条目。
        否则至少返回一个最小数量，但若存在同排名条目，
        实际返回数量可能更多，不过不会超过最大返回数量。

        重复路线会被忽略。

        条目依据排名进行比较。
        单目标分析中，排名实际就是分数，越高越好；
        多目标分析中，排名是帕累托等级，越小越好。

        :param items: 已按排名排序、待筛选的超节点或反应树
        :param scores: 各节点对应的分数
        :param reaction: 每个条目对应的反应序列，用于检查重复
        :param ranks: 每个条目的排名，用于决定选择范围
        :param selection: 最小/最大返回数，或是否返回全部的指示
        :param min_comp: 若为 `True` 表示排名越高越好，否则表示越低越好
        """
        if len(items) <= selection.nmin:
            return items, scores

        max_return, min_return = selection.nmax, selection.nmin
        if selection.return_all:
            nsolved = sum(int(item.is_solved) for item in items)
            if nsolved:
                max_return = nsolved
                min_return = nsolved

        seen_hashes = set()
        best_items: List[Any] = []
        best_scores = []
        if min_comp:
            last_rank = 1e16
            comp_op = operator.lt
        else:
            last_rank = 0
            comp_op = operator.gt
        for rank, score, item, actions in zip(ranks, scores, items, reactions):
            if len(best_items) >= min_return and comp_op(rank, last_rank):
                break
            route_hash = hash_reactions(actions)

            if route_hash in seen_hashes:
                continue
            seen_hashes.add(route_hash)
            best_items.append(item)
            best_scores.append(score)
            last_rank = rank

            if max_return and len(best_items) == max_return:
                break

        return best_items, best_scores

    @staticmethod
    def _policy_used_statistics(
        reactions: Iterable[Union[RetroReaction, FixedRetroReaction]]
    ) -> StrDict:
        policy_used_counts: StrDict = defaultdict(lambda: 0)
        for reaction in reactions:
            policy_used = reaction.metadata.get("policy_name")
            if policy_used:
                policy_used_counts[policy_used] += 1
        return dict(policy_used_counts)

    @staticmethod
    def _top_ranked_join(items: Iterable[Any]) -> Union[str, Any]:
        items = list(items)
        if len(items) == 1:
            return items[0]
        return "|".join(f"{item}" for item in items)
