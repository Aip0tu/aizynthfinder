"""定义搜索树节点的模块。"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from paretoset import paretoset

from aizynthfinder.chem import TreeMolecule, deserialize_action, serialize_action
from aizynthfinder.search.mcts.state import MctsState
from aizynthfinder.search.mcts.utils import ReactionTreeFromSuperNode, route_to_node
from aizynthfinder.utils.exceptions import (
    NodeUnexpectedBehaviourException,
    RejectionException,
)
from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.chem import (
        MoleculeDeserializer,
        MoleculeSerializer,
        RetroReaction,
    )
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.reactiontree import ReactionTree
    from aizynthfinder.search.mcts.search import MctsSearchTree
    from aizynthfinder.utils.type_utils import List, Optional, StrDict, Tuple


class MctsNode:
    """
    搜索树中的单个节点。

    为了提升效率，子节点采用惰性实例化：
    只有当某个子节点被选中时，才会真正应用对应反应并创建该子节点。

    对于已经实例化的子节点，可以通过以下方式访问其属性：

    .. code-block::

        children_attr = node[child]

    返回值是一个字典，包含 `"action"`、`"value"`、`"prior"` 和
    `"visitations"` 等键。

    :ivar is_expanded: 节点是否已经生成过子节点
    :ivar is_expandable: 节点是否仍可继续扩展
    :ivar tree: 拥有该节点的搜索树

    :param state: 节点状态
    :param owner: 持有该节点的搜索树
    :param config: 树搜索算法配置
    :param parent: 父节点，默认为 `None`
    """

    def __init__(
        self,
        state: MctsState,
        owner: MctsSearchTree,
        config: Configuration,
        parent: Optional[MctsNode] = None,
    ):
        self._state = state
        self._config = config
        self._expansion_policy = config.expansion_policy
        self._filter_policy = config.filter_policy
        self.tree = owner
        self.is_expanded: bool = False
        self.is_expandable: bool = not self.state.is_terminal
        self._parent = parent

        if owner is None:
            self.created_at_iteration: Optional[int] = None
        else:
            self.created_at_iteration = self.tree.profiling["iterations"]

        self._children_values: List[float] = []
        self._children_priors: List[float] = []
        self._children_visitations: List[int] = []
        self._children_actions: List[RetroReaction] = []
        self._children: List[Optional[MctsNode]] = []

        self.blacklist = set(mol.inchi_key for mol in state.expandable_mols)
        if parent:
            self.blacklist = self.blacklist.union(parent.blacklist)

        if self._algo_config["mcts_grouping"]:
            self._degeneracy_check = self._algo_config["mcts_grouping"].lower()
        else:
            self._degeneracy_check = "none"
        self._logger = logger()

    def __getitem__(self, node: "MctsNode") -> StrDict:
        idx = self._children.index(node)
        return {
            "action": self._children_actions[idx],
            "value": self._children_values[idx],
            "prior": self._children_priors[idx],
            "visitations": self._children_visitations[idx],
        }

    @classmethod
    def create_root(
        cls, smiles: str, tree: MctsSearchTree, config: Configuration
    ) -> "MctsNode":
        """
        使用给定的 SMILES 创建树根节点。

        :param smiles: 根状态对应的 SMILES 表示
        :param tree: 搜索树
        :param config: 树搜索算法配置
        :return: 创建好的根节点
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        state = MctsState(mols=[mol], config=config)
        return cls(state=state, owner=tree, config=config)

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        tree: MctsSearchTree,
        config: Configuration,
        molecules: MoleculeDeserializer,
        parent: Optional["MctsNode"] = None,
    ) -> "MctsNode":
        """
        从字典创建新节点，即执行反序列化。

        :param dict_: 序列化后的节点字典
        :param tree: 搜索树
        :param config: 树搜索算法配置
        :param molecules: 已反序列化的分子对象
        :param parent: 父节点
        :return: 反序列化得到的节点
        """
        # pylint: disable=protected-access
        state = MctsState.from_dict(dict_["state"], config, molecules)
        node = cls(state=state, owner=tree, config=config, parent=parent)
        node.is_expanded = dict_["is_expanded"]
        node.is_expandable = dict_["is_expandable"]
        node._children_values = dict_["children_values"]
        node._children_priors = dict_["children_priors"]
        node._children_visitations = dict_["children_visitations"]
        node._children_actions = [
            deserialize_action(action_dict, molecules)
            for action_dict in dict_["children_actions"]
        ]
        node._children = [
            cls.from_dict(child, tree, config, molecules, parent=node)
            if child
            else None
            for child in dict_["children"]
        ]
        return node

    @property
    def children(self) -> List["MctsNode"]:
        """
        返回所有已经实例化的子节点。

        :return: 子节点列表
        """
        return [child for child in self._children if child]

    @property
    def is_solved(self) -> bool:
        """返回当前状态是否已求解。"""
        return self.state.is_solved

    @property
    def parent(self) -> Optional["MctsNode"]:
        """返回当前节点的父节点。"""
        return self._parent

    @property
    def state(self) -> MctsState:
        """返回节点内部维护的状态对象。"""
        return self._state

    @property
    def _algo_config(self) -> StrDict:
        """为算法配置提供一个更简短的访问入口。"""
        return self._config.search.algorithm_config

    def actions_to(self) -> List[RetroReaction]:
        """
        返回通往当前节点的动作序列。

        :return: 动作列表
        """
        return self.path_to()[0]

    def backpropagate(self, child: "MctsNode", value_estimate: float) -> None:
        """
        更新某个子节点的访问次数及累计价值。

        :param child: 子节点
        :param value_estimate: 要累加到子节点价值上的估计值
        """
        idx = self._children.index(child)
        self._children_visitations[idx] += 1
        self._children_values[idx] += value_estimate

    def children_view(self) -> StrDict:
        """
        创建子节点属性的只读视图。

        返回字典中的各个列表都会重新创建，
        但实际的子节点对象并不会被复制。

        返回字典包含 `"actions"`、`"values"`、`"priors"`、
        `"visitations"` 和 `"objects"` 等键。

        :return: 子节点属性视图
        """
        return {
            "actions": list(self._children_actions),
            "values": list(self._children_values),
            "priors": list(self._children_priors),
            "visitations": list(self._children_visitations),
            "objects": list(self._children),
        }

    def expand(self) -> None:
        """
        展开当前节点。

        展开指的是为当前节点生成候选子节点信息，
        但此时并不立即实例化真正的子节点对象。
        动作及其先验概率来自策略网络。

        不过，如果某些策略被标记为需要立即实例化，
        则对应的子节点会在这里直接创建出来。
        """
        if self.is_expanded:
            msg = f"Oh no! This node is already expanded. id={id(self)}"
            self._logger.debug(msg)
            raise NodeUnexpectedBehaviourException(msg)

        if self.is_expanded or not self.is_expandable:
            return

        self.is_expanded = True

        cache_molecules = []
        if self.parent:
            for child in self.parent.children:
                if child is not self:
                    cache_molecules.extend(child.state.expandable_mols)

        # 计算可选动作，并填充子节点信息列表。
        # 默认情况下，一个动作只假设对应一组反应物。
        actions, priors = self._expansion_policy(
            self.state.expandable_mols, cache_molecules
        )
        self._fill_children_lists(actions, priors)

        # 如果扩展没有生成任何子节点，则撤销展开状态。
        if len(actions) == 0:
            self.is_expandable = False
            self.is_expanded = False

        if self.tree:
            self.tree.profiling["expansion_calls"] += 1

        if not self._algo_config["immediate_instantiation"]:
            return
        # 对被标记的策略产出的所有子动作执行立即实例化。
        # 这里遍历的是动作列表切片，因为实例化过程中列表可能继续增长。
        nactions = len(actions)
        for child_idx, action in enumerate(self._children_actions[:nactions]):
            policy_name = action.metadata.get("policy_name")
            if (
                policy_name
                and policy_name in self._algo_config["immediate_instantiation"]
            ):
                self._instantiate_child(child_idx)

    def is_terminal(self) -> bool:
        """
        节点在两种情况下会被视为终止节点：
        当前不可继续扩展，或其内部状态本身已经终止（已求解）。

        :return: 节点是否终止
        """
        return not self.is_expandable or self.state.is_terminal

    def path_to(self) -> Tuple[List[RetroReaction], List[MctsNode]]:
        """
        返回到达当前节点的路径。

        路径由动作列表和节点列表组成。

        :return: 动作列表与节点列表
        """
        return route_to_node(self)

    def promising_child(self) -> Optional["MctsNode"]:
        """
        返回当前 `Q + U` 值最高的子节点。

        如果该子节点尚未实例化，会先完成实例化。

        如果找不到任何可用动作，则返回 `None`。

        :return: 选中的子节点
        """
        child = None
        while child is None:
            try:
                child = self._score_and_select()
            # `_score_and_select` 在无可选子节点时会抛出异常。
            except ValueError:
                child = None
                break

        if not child:
            self._logger.debug(
                "Returning None from promising_child() because there were no applicable action"
            )
            self.is_expanded = False
            self.is_expandable = False

        return child

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        将节点对象序列化为字典。

        :param molecule_store: 分子序列化存储器
        :return: 序列化后的节点字典
        """
        return {
            "state": self.state.serialize(molecule_store),
            "children_values": self._serialize_stats_list("_children_values"),
            "children_priors": self._serialize_stats_list("_children_priors"),
            "children_visitations": self._children_visitations,
            "children_actions": [
                serialize_action(action, molecule_store)
                for action in self._children_actions
            ],
            "children": [
                child.serialize(molecule_store) if child else None
                for child in self._children
            ],
            "is_expanded": self.is_expanded,
            "is_expandable": self.is_expandable,
        }

    def to_reaction_tree(self) -> ReactionTree:
        """
        根据通往当前节点的动作和节点路径构建反应树。

        :return: 构建得到的反应树
        """
        return ReactionTreeFromSuperNode(self).tree

    def _check_child_reaction(self, reaction: RetroReaction) -> bool:
        if not reaction.reactants:
            self._logger.debug(f"{reaction} did not produce any reactants")
            return False

        # fmt: off
        reactants0 = reaction.reactants[0]
        if len(reaction.reactants) == 1 and len(reactants0) == 1 and reaction.mol == reactants0[0]:
            return False
        # fmt: on

        return True

    def _children_q(self) -> np.ndarray:
        return np.array(self._children_values) / np.array(self._children_visitations)

    def _children_u(self) -> np.ndarray:
        total_visits = np.log(np.sum(self._children_visitations))
        child_visits = np.array(self._children_visitations)
        return self._algo_config["C"] * np.sqrt(2 * total_visits / child_visits)

    def _create_children_nodes(
        self, states: List[MctsState], child_idx: int
    ) -> List["MctsNode"]:
        new_nodes = []
        first_child_idx = child_idx
        for state_index, state in enumerate(states):
            if self._generated_degeneracy(state, first_child_idx):
                # 只需禁用第一个新子节点；
                # 如果该动作生成了更多状态，则直接跳过这些状态的子节点创建。
                if state_index == 0:
                    self._disable_child(child_idx)
                continue

            # 如果动作有多个产物结果，需要同步扩展各个子节点信息列表。
            if state_index > 0:
                child_idx = self._expand_children_lists(first_child_idx, state_index)

            if self._filter_child_reaction(self._children_actions[child_idx]):
                self._disable_child(child_idx)
            else:
                new_node = self.__class__(
                    state=state, owner=self.tree, config=self._config, parent=self
                )
                self._children[child_idx] = new_node
                new_nodes.append(new_node)
        return new_nodes

    def _disable_child(self, child_idx: int) -> None:
        self._children_values[child_idx] = -1e6

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        new_action = self._children_actions[old_index].copy(index=action_index)
        self._children_actions.append(new_action)
        self._children_priors.append(self._children_priors[old_index])
        self._children_values.append(self._children_values[old_index])
        self._children_visitations.append(self._children_visitations[old_index])
        self._children.append(None)
        return len(self._children) - 1

    def _fill_children_lists(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> None:
        self._children_actions = actions
        self._children_priors = priors
        nactions = len(actions)
        self._children_visitations = [1] * nactions
        self._children = [None] * nactions
        if self._algo_config["use_prior"]:
            self._children_values = list(self._children_priors)
        else:
            self._children_values = [self._algo_config["default_prior"]] * nactions

    def _filter_child_reaction(self, reaction: RetroReaction) -> bool:
        if self._regenerated_blacklisted(reaction):
            self._logger.debug(
                f"Reaction {reaction.reaction_smiles()} "
                f"was rejected because it re-generated molecule not in stock"
            )
            return True

        if not self._filter_policy.selection:
            return False
        try:
            self._filter_policy(reaction)
        except RejectionException as err:
            self._logger.debug(str(err))
            return True
        return False

    def _generated_degeneracy(self, new_state: MctsState, child_idx: int) -> bool:
        """
        检查新的 MCTS 状态是否与某个子节点的状态重复。

        检查方式可以是 `"partial"`，只比较可扩展分子；
        也可以是 `"full"`，比较状态中的全部分子。

        尚未展开的子节点和终止子节点不会参与比较。

        如果判定为重复，新动作的元数据会附加到此前已创建的等价状态动作上。
        """

        def equal_states(query_state):
            if self._degeneracy_check == "partial":
                return query_state.expandables_hash == new_state.expandables_hash
            return query_state == new_state

        if self._degeneracy_check not in ["partial", "full"]:
            return False
        previous_action = None
        for child, action in zip(self._children, self._children_actions):
            if (
                child is not None
                and not child.is_terminal()
                and equal_states(child.state)
            ):
                previous_action = action
                break

        if previous_action is None:
            return False

        # 同一动作对象的元数据本来就是共享的，因此无需重复拷贝。
        if previous_action is self._children_actions[child_idx]:
            return True

        metadata_copy = dict(self._children_actions[child_idx].metadata)
        if "additional_actions" not in previous_action.metadata:
            previous_action.metadata["additional_actions"] = []
        previous_action.metadata["additional_actions"].append(metadata_copy)
        return True

    def _instantiate_child(self, child_idx: int) -> List["MctsNode"]:
        """
        实例化指定的子节点。

        算法流程如下：
        * 应用该子节点对应的反应
        * 如果反应应用失败，则将其值设为 `-1e6` 并返回空结果
        * 为每个反应结果创建一个新状态
        * 创建新的子节点
        * 如果过滤策略判断某个结果不可行，则将对应子节点的值设为 `-1e6`
        * 返回所有新建节点
        """
        if self._children[child_idx] is not None:
            raise NodeUnexpectedBehaviourException("Node already instantiated")

        reaction = self._children_actions[child_idx]
        if reaction.unqueried:
            if self.tree:
                self.tree.profiling["reactants_generations"] += 1
            _ = reaction.reactants

        if not self._check_child_reaction(reaction):
            self._disable_child(child_idx)
            return []

        keep_mols = [mol for mol in self.state.mols if mol is not reaction.mol]
        new_states = [
            MctsState(keep_mols + list(reactants), self._config)
            for reactants in reaction.reactants
        ]
        return self._create_children_nodes(new_states, child_idx)

    def _regenerated_blacklisted(self, reaction: RetroReaction) -> bool:
        if not self._algo_config["prune_cycles_in_search"]:
            return False
        for reactants in reaction.reactants:
            for mol in reactants:
                if mol.inchi_key in self.blacklist:
                    return True
        return False

    def _score_and_select(self) -> Optional["MctsNode"]:
        if not max(self._children_values) > 0:
            raise ValueError("Has no selectable children")
        scores = self._children_q() + self._children_u()
        indices = np.where(scores == scores.max())[0]
        index = np.random.choice(indices)
        return self._select_child(index)

    def _select_child(self, child_idx: int) -> Optional["MctsNode"]:
        """
        选择某个子节点时，必要时会触发其实例化。

        如果该子节点已存在，则直接返回；
        否则会在新创建出的可行节点中随机返回一个。
        """
        if self._children[child_idx]:
            return self._children[child_idx]

        new_nodes = self._instantiate_child(child_idx)
        if new_nodes:
            return random.choice(new_nodes)
        return None

    def _serialize_stats_list(self, name: str) -> List[float]:
        return [float(value) for value in getattr(self, name)]


class ParetoMctsNode(MctsNode):
    """
    多目标树搜索中的节点。

    该实现基于以下算法：
        Chen W., Liu L. Pareto Monte Carlo Tree Search for Multi-Objective Informative Planning
        Robotics: Science and Systems 2019, 2012 arXiv:2111.01825


    与标准 MCTS 相比，主要差异在于：
        - 子节点统计量中的价值、累计奖励和先验值都是嵌套列表，
          每个目标各占一个值
        - 选择阶段会先计算帕累托前沿，再从其中随机选取一个子节点

    在本实现中，子节点一旦被访问过一次，就不再继续考虑其先验项。

    默认假设所有目标都需要最大化。
    """

    def __init__(
        self,
        state: MctsState,
        owner: MctsSearchTree,
        config: Configuration,
        parent: Optional[ParetoMctsNode] = None,
    ):
        super().__init__(state, owner, config, parent)
        self._num_objectives = len(self._algo_config["search_rewards"])
        self._prior_weight = 1
        self._direction = "max"  # 当前实现默认所有目标都需要最大化。
        self._children_rewards_cummulative: List[List[float]]
        self._children_values: List[List[float]]  # type: ignore
        self._children_priors: List[List[float]]  # type: ignore

    def backpropagate(self, child: "MctsNode", value_estimate: List[float]) -> None:  # type: ignore
        """
        更新某个子节点的访问次数及累计价值。

        :param child: 子节点
        :param value_estimate: 要累加到子节点价值上的估计值
        """
        idx = self._children.index(child)
        self._children_visitations[idx] += 1
        # 这里只更新累计奖励，
        # `_children_values` 会在选择阶段再同步刷新。
        new_value_estimate = [
            cum_reward + new_reward
            for cum_reward, new_reward in zip(
                self._children_rewards_cummulative[idx], value_estimate
            )
        ]
        self._children_rewards_cummulative[idx] = new_value_estimate

    def children_view(self) -> StrDict:
        """
        创建子节点属性的只读视图。

        返回的各个列表都是新建副本，但子节点对象本身不会被复制。

        返回字典包含 `"actions"`、`"values"`、`"priors"`、
        `"visitations"`、`"rewards_cum"` 和 `"objects"` 等键。

        :return: 子节点属性视图
        """
        dict_ = super().children_view()
        dict_["rewards_cum"] = list(self._children_rewards_cummulative)
        return dict_

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        将节点对象序列化为字典。

        :param molecule_store: 分子序列化存储器
        :return: 序列化后的节点字典
        """
        dict_ = super().serialize(molecule_store)
        dict_["children_cumulative_reward"] = self._serialize_stats_list(
            "_children_rewards_cummulative"
        )
        return dict_

    def _disable_child(self, child_idx: int) -> None:
        self._children_rewards_cummulative[child_idx] = [-1e6] * self._num_objectives

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        ret = super()._expand_children_lists(old_index, action_index)
        # 这些字段是嵌套列表，因此也要为内部列表创建独立副本。
        self._children_values[-1] = list(self._children_values[-1])
        self._children_priors[-1] = list(self._children_priors[-1])
        self._children_rewards_cummulative.append(
            list(self._children_rewards_cummulative[old_index])
        )
        return ret

    def _fill_children_lists(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> None:
        self._children_actions = actions
        nactions = len(actions)
        # 形状：num_actions x 1
        self._children_visitations = [1] * nactions
        self._children = [None] * nactions
        # 形状：num_actions x num_objectives
        self._children_rewards_cummulative = [[0.0] * self._num_objectives] * nactions
        if self._algo_config["use_prior"]:
            # 形状：num_actions x num_objectives
            # 例如子节点 i 对应 3 个目标时，先验会展开为 [prior_i, prior_i, prior_i]。
            self._children_priors = [[prior] * self._num_objectives for prior in priors]

        else:
            self._children_priors = [
                [self._algo_config["default_prior"]] * self._num_objectives
            ] * nactions

        # 初始化时累计奖励为零，因此价值先等于先验项。
        self._children_values = [
            [prior * self._prior_weight for prior in priors]
            for priors in self._children_priors
        ]

    def _children_q(self, children_values_arr):
        children_visitations_expanded = np.repeat(
            np.array(self._children_visitations).reshape(-1, 1),
            axis=1,
            repeats=self._num_objectives,
        )
        return children_values_arr / children_visitations_expanded

    def _compute_children_scores(self) -> np.ndarray:
        """计算修正后的 UCB 分数：`alpha * prior + average reward + exploration`。"""
        # 节点一旦被访问过，就把其先验项衰减为零。
        children_priors_arr = self._prior_schedule_oneoff()
        # 计算 `prior_weight * prior + cumulative_rewards`。
        children_values_arr = self._prior_weight * children_priors_arr + np.array(
            self._children_rewards_cummulative
        )
        expanded_u = np.repeat(
            self._children_u().reshape(-1, 1), axis=1, repeats=self._num_objectives
        )
        # `_children_scores` 的形状为 num_children x num_objectives。
        children_scores = self._children_q(children_values_arr) + expanded_u
        if children_scores.shape[1] != self._num_objectives:
            raise ValueError(
                f"expected second dimension to have {self._num_objectives},"
                f"currently has {children_scores.shape[1]}"
            )
        self._children_values = children_values_arr.tolist()
        self._children_priors = children_priors_arr.tolist()
        return children_scores

    def _prior_schedule_oneoff(self) -> np.ndarray:
        # 形状：num_children x 1
        visted_mask = (np.array(self._children_visitations) > 1).reshape(-1, 1)
        # 形状：num_children x num_objectives
        visted_mask = np.repeat(visted_mask, axis=1, repeats=self._num_objectives)
        # 已访问子节点的先验权重会被置零。
        children_priors_arr = np.array(self._children_priors)
        children_priors_arr[visted_mask] = 0
        return children_priors_arr

    def _score_and_select(self) -> Optional["MctsNode"]:
        if not max(max(value_list) for value_list in self._children_values) > 0:
            raise ValueError("Has no selectable children")
        children_scores = self._compute_children_scores()
        pareto_idxs = self._update_pareto_front(children_scores)
        index = np.random.choice(pareto_idxs)
        return self._select_child(index)

    def _serialize_stats_list(self, name: str):
        return [
            [float(value) for value in value_list] for value_list in getattr(self, name)
        ]

    def _update_pareto_front(self, children_scores: np.ndarray) -> np.ndarray:
        """
        更新节点的帕累托前沿。

        这一步通常发生在最佳子节点的价值更新之后。

        :param children_scores: 子节点分数
        :returns: 位于帕累托前沿的子节点索引
        """
        direction_arr = np.repeat(self._direction, self._num_objectives)
        mask = paretoset(children_scores, sense=direction_arr, distinct=False)
        return np.arange(len(self._children))[mask]
