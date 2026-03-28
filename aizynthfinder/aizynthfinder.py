"""包含逆合成工具主入口类的模块。"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

from tqdm import tqdm

from aizynthfinder.analysis import (
    RouteCollection,
    RouteSelectionArguments,
    TreeAnalysis,
)
from aizynthfinder.chem import FixedRetroReaction, Molecule, TreeMolecule
from aizynthfinder.context.config import Configuration
from aizynthfinder.context.policy import BondFilter
from aizynthfinder.context.scoring import BrokenBondsScorer, CombinedScorer
from aizynthfinder.reactiontree import ReactionTreeFromExpansion
from aizynthfinder.search.andor_trees import AndOrSearchTreeBase
from aizynthfinder.search.mcts import MctsSearchTree
from aizynthfinder.utils.exceptions import MoleculeException
from aizynthfinder.utils.loading import load_dynamic_class

# 必须最先导入，用于配置 rdkit、tensorflow 等依赖的日志。
from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.utils.type_utils import (
        Callable,
        Dict,
        List,
        Optional,
        StrDict,
        Tuple,
        Union,
    )


class AiZynthFinder:
    """
    `aizynthfinder` 工具的公开 API。

    如果传入 yaml 配置文件路径或配置字典进行实例化，
    库存与策略网络会被直接加载。
    否则，用户需要在执行树搜索之前自行完成这些资源的加载。

    :ivar config: 搜索配置
    :ivar expansion_policy: 扩展策略模型
    :ivar filter_policy: 过滤策略模型
    :ivar stock: 库存对象
    :ivar scorers: 已加载的评分器
    :ivar tree: 搜索树
    :ivar analysis: 树分析对象
    :ivar routes: 排名靠前的路线集合
    :ivar search_stats: 最近一次搜索的统计信息

    :param configfile: yaml 配置文件路径，优先级高于 `configdict`，默认为 `None`
    :param configdict: 以字典形式传入的配置，默认为 `None`
    """

    def __init__(
        self, configfile: Optional[str] = None, configdict: Optional[StrDict] = None
    ) -> None:
        self._logger = logger()

        if configfile:
            self.config = Configuration.from_file(configfile)
        elif configdict:
            self.config = Configuration.from_dict(configdict)
        else:
            self.config = Configuration()

        self.expansion_policy = self.config.expansion_policy
        self.filter_policy = self.config.filter_policy
        self.stock = self.config.stock
        self.scorers = self.config.scorers
        self.tree: Optional[Union[MctsSearchTree, AndOrSearchTreeBase]] = None
        self._target_mol: Optional[Molecule] = None
        self.search_stats: StrDict = dict()
        self.routes = RouteCollection([])
        self.analysis: Optional[TreeAnalysis] = None
        self._num_objectives = len(
            self.config.search.algorithm_config.get("search_rewards", [])
        )

    @property
    def target_smiles(self) -> str:
        """返回待预测路线目标分子的 SMILES 表示。"""
        if not self._target_mol:
            return ""
        return self._target_mol.smiles

    @target_smiles.setter
    def target_smiles(self, smiles: str) -> None:
        self.target_mol = Molecule(smiles=smiles)

    @property
    def target_mol(self) -> Optional[Molecule]:
        """返回待进行路线预测的目标分子。"""
        return self._target_mol

    @target_mol.setter
    def target_mol(self, mol: Molecule) -> None:
        self.tree = None
        self._target_mol = mol

    def build_routes(
        self,
        selection: Optional[RouteSelectionArguments] = None,
        scorer: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        构建反应路线。

        树搜索完成后，需要调用此方法才能从搜索树中提取路线结果。

        :param selection: 路线筛选条件
        :param scorer: 用于给节点打分的对象引用，也可以是列表
        :raises ValueError: 当搜索树尚未初始化时抛出
        """
        self.analysis = self._setup_analysis(scorer=scorer)
        config_selection = RouteSelectionArguments(
            nmin=self.config.post_processing.min_routes,
            nmax=self.config.post_processing.max_routes,
            return_all=self.config.post_processing.all_routes,
        )
        self.routes = RouteCollection.from_analysis(
            self.analysis, selection or config_selection
        )

    def extract_statistics(self) -> StrDict:
        """以字典形式提取搜索树统计信息。"""
        if not self.analysis:
            return {}
        stats = {
            "target": self.target_smiles,
            "search_time": self.search_stats["time"],
            "first_solution_time": self.search_stats.get("first_solution_time", 0),
            "first_solution_iteration": self.search_stats.get(
                "first_solution_iteration", 0
            ),
        }
        stats.update(self.analysis.tree_statistics())
        return stats

    def prepare_tree(self) -> None:
        """
        为搜索过程准备树结构。

        :raises ValueError: 当目标分子尚未设置时抛出
        """
        if not self.target_mol:
            raise ValueError("No target molecule set")

        try:
            self.target_mol.sanitize()
        except MoleculeException:
            raise ValueError("Target molecule unsanitizable")

        self.stock.reset_exclusion_list()
        if (
            self.config.search.exclude_target_from_stock
            and self.target_mol in self.stock
        ):
            self.stock.exclude(self.target_mol)
            self._logger.debug("Excluding the target compound from the stock")

        if self.config.search.break_bonds or self.config.search.freeze_bonds:
            self._setup_focussed_bonds(self.target_mol)

        self._setup_search_tree()
        self.analysis = None
        self.routes = RouteCollection([])
        self.filter_policy.reset_cache()
        self.expansion_policy.reset_cache()

    def stock_info(self) -> StrDict:
        """
        返回已收集反应树中所有叶子节点的库存可用性。

        返回字典的键是叶子节点的 SMILES 字符串，
        值是对应的库存可用性信息。

        :return: 收集得到的库存信息
        """
        if not self.analysis:
            return {}
        _stock_info = {}
        for tree in self.routes.reaction_trees:
            for leaf in tree.leafs():
                if leaf.smiles not in _stock_info:
                    _stock_info[leaf.smiles] = self.stock.availability_list(leaf)
        return _stock_info

    def tree_search(self, show_progress: bool = False) -> float:
        """
        执行实际的树搜索。

        :param show_progress: 若为 `True`，显示进度条
        :return: 搜索耗时，单位为秒
        """
        if not self.tree:
            self.prepare_tree()
        # 这里只是为了通过类型检查，`prepare_tree()` 会负责创建树。
        assert self.tree is not None
        self.search_stats = {"returned_first": False, "iterations": 0}

        time0 = time.time()
        i = 1
        self._logger.debug("Starting search")
        time_past = time.time() - time0

        if show_progress:
            pbar = tqdm(total=self.config.search.iteration_limit, leave=False)

        while (
            time_past < self.config.search.time_limit
            and i <= self.config.search.iteration_limit
        ):
            if show_progress:
                pbar.update(1)
            self.search_stats["iterations"] += 1

            try:
                is_solved = self.tree.one_iteration()
            except StopIteration:
                break

            if is_solved and "first_solution_time" not in self.search_stats:
                self.search_stats["first_solution_time"] = time.time() - time0
                self.search_stats["first_solution_iteration"] = i

            if self.config.search.return_first and is_solved:
                self._logger.debug("Found first solved route")
                self.search_stats["returned_first"] = True
                break
            i = i + 1
            time_past = time.time() - time0

        if show_progress:
            pbar.close()
        time_past = time.time() - time0
        self._logger.debug("Search completed")
        self.search_stats["time"] = time_past
        return time_past

    def _setup_focussed_bonds(self, target_mol: Molecule) -> None:
        """
        配置多目标评分函数中的 “broken bonds” 评分器，
        并向过滤策略中添加 “frozen bonds” 过滤器。

        :param target_mol: 目标分子
        """
        target_mol = TreeMolecule(smiles=target_mol.smiles, parent=None)

        bond_filter_key = "__finder_bond_filter"
        if self.config.search.freeze_bonds:
            if not target_mol.has_all_focussed_bonds(self.config.search.freeze_bonds):
                raise ValueError("Bonds in 'freeze_bond' must exist in target molecule")
            bond_filter = BondFilter(bond_filter_key, self.config)
            self.filter_policy.load(bond_filter)
            self.filter_policy.select(bond_filter_key, append=True)
        elif (
            self.filter_policy.selection
            and bond_filter_key in self.filter_policy.selection
        ):
            self.filter_policy.deselect(bond_filter_key)

        search_rewards = self.config.search.algorithm_config.get("search_rewards")
        if not search_rewards:
            return

        if self.config.search.break_bonds and "broken bonds" in search_rewards:
            if not target_mol.has_all_focussed_bonds(self.config.search.break_bonds):
                raise ValueError("Bonds in 'break_bonds' must exist in target molecule")
            self.scorers.load(BrokenBondsScorer(self.config))
            self._num_objectives = len(search_rewards)

    def _setup_search_tree(self) -> None:
        """根据当前配置初始化搜索树实例。"""
        self._logger.debug(f"Defining tree root:  {self.target_smiles}")
        if self.config.search.algorithm.lower() == "mcts":
            self.tree = MctsSearchTree(
                root_smiles=self.target_smiles, config=self.config
            )
        else:
            cls = load_dynamic_class(self.config.search.algorithm)
            self.tree = cls(root_smiles=self.target_smiles, config=self.config)

    def _setup_analysis(
        self,
        scorer: Optional[Union[str, List[str]]],
    ) -> TreeAnalysis:
        """配置 `TreeAnalysis` 实例。

        :param scorer: 用于给节点打分的对象引用，也可以是列表
        :returns: 配置完成的 `TreeAnalysis`
        :raises ValueError: 当搜索树尚未初始化时抛出
        """
        if not self.tree:
            raise ValueError("Search tree not initialized")

        if scorer is None:
            scorer_names = self.config.post_processing.route_scorers
            # 若未显式指定，则复用搜索阶段的奖励评分器。
            if not scorer_names:
                search_rewards = self.config.search.algorithm_config.get(
                    "search_rewards"
                )
                scorer_names = search_rewards if search_rewards else ["state score"]

        elif isinstance(scorer, str):
            scorer_names = [scorer]
        else:
            scorer_names = list(scorer)

        if "broken bonds" in scorer_names:
            # 按需补充 broken bonds 评分器。
            self.scorers.load(BrokenBondsScorer(self.config))

        scorers = [self.scorers[name] for name in scorer_names]

        if self.config.post_processing.scorer_weights:
            scorers = [
                CombinedScorer(
                    self.config,
                    scorer_names,
                    self.config.post_processing.scorer_weights,
                )
            ]

        return TreeAnalysis(self.tree, scorers)


class AiZynthExpander:
    """
    AiZynthFinder 扩展策略与过滤策略的公开 API。

    如果传入 yaml 配置文件路径或配置字典进行实例化，
    库存与策略网络会被直接加载。
    否则，用户需要在执行树搜索之前自行完成这些资源的加载。

    :ivar config: 搜索配置
    :ivar expansion_policy: 扩展策略模型
    :ivar filter_policy: 过滤策略模型

    :param configfile: yaml 配置文件路径，优先级高于 `configdict`，默认为 `None`
    :param configdict: 以字典形式传入的配置，默认为 `None`
    """

    def __init__(
        self, configfile: Optional[str] = None, configdict: Optional[StrDict] = None
    ) -> None:
        self._logger = logger()

        if configfile:
            self.config = Configuration.from_file(configfile)
        elif configdict:
            self.config = Configuration.from_dict(configdict)
        else:
            self.config = Configuration()

        self.expansion_policy = self.config.expansion_policy
        self.filter_policy = self.config.filter_policy
        self.stats: StrDict = {}

    def do_expansion(
        self,
        smiles: str,
        return_n: int = 5,
        filter_func: Optional[Callable[[RetroReaction], bool]] = None,
    ) -> List[Tuple[FixedRetroReaction, ...]]:
        """
        对给定分子执行扩展，并返回反应元组列表。

        返回列表中的每个元组都包含生成相同反应物的一组反应，
        因而这种嵌套结构也承担了反应分组的作用。

        如果已配置过滤策略，会将反应的可行性概率添加到反应元数据中。

        额外的过滤函数可用于实现自定义筛选。
        该可调用对象应仅接收一个 `RetroReaction` 参数；
        返回 `True` 表示保留该反应，返回 `False` 表示移除该反应。

        :param smiles: 目标分子的 SMILES 字符串
        :param return_n: 返回列表的最大长度
        :param filter_func: 额外的过滤函数
        :return: 按反应物分组后的反应集合
        """
        self.stats = {"non-applicable": 0}

        mol = TreeMolecule(parent=None, smiles=smiles)
        actions, _ = self.expansion_policy.get_actions([mol])
        results: Dict[Tuple[str, ...], List[FixedRetroReaction]] = defaultdict(list)
        for action in actions:
            reactants = action.reactants
            if not reactants:
                self.stats["non-applicable"] += 1
                continue
            if filter_func and not filter_func(action):
                continue
            for name in self.filter_policy.selection or []:
                if hasattr(self.filter_policy[name], "feasibility"):
                    _, feasibility_prob = self.filter_policy[name].feasibility(action)
                    action.metadata["feasibility"] = float(feasibility_prob)
                    break
            action.metadata["expansion_rank"] = len(results) + 1
            unique_key = tuple(sorted(mol.inchi_key for mol in reactants[0]))
            if unique_key not in results and len(results) >= return_n:
                continue
            rxn = next(ReactionTreeFromExpansion(action).tree.reactions())  # type: ignore
            results[unique_key].append(rxn)
        return [tuple(reactions) for reactions in results.values()]
