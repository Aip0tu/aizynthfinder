"""AND/OR 搜索树基础类与辅助工具。"""
from __future__ import annotations

import abc
import random
from typing import TYPE_CHECKING

import networkx as nx

from aizynthfinder.chem import UniqueMolecule
from aizynthfinder.reactiontree import ReactionTree, ReactionTreeLoader

if TYPE_CHECKING:
    from aizynthfinder.chem import FixedRetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.context.stock import Stock
    from aizynthfinder.utils.type_utils import Any, List, Optional, StrDict, Union


class TreeNodeMixin:
    """树节点通用混入类。"""

    @property
    def prop(self) -> StrDict:
        """返回对外暴露的属性字典。"""
        return {}

    @property
    def children(self) -> List["TreeNodeMixin"]:
        """返回子节点列表。"""
        return []


class AndOrSearchTreeBase(abc.ABC):
    """基于 AND/OR 结构的搜索树基类。"""

    def __init__(
        self, config: Configuration, root_smiles: Optional[str] = None
    ) -> None:
        self.config = config
        self._root_smiles = root_smiles

    @property
    def mol_nodes(self) -> List[TreeNodeMixin]:
        """返回树中的分子节点列表。"""
        return []

    @abc.abstractmethod
    def one_iteration(self) -> bool:
        """执行一次搜索迭代。"""
        return False

    @abc.abstractmethod
    def routes(self) -> List[ReactionTree]:
        """返回树中提取出的路线。"""
        return []


class SplitAndOrTree:
    """
    将 AND/OR 树拆分为多条独立路线的算法封装。

    这是对 CompRet 论文中算法的一个改写版本：
    Shibukawa et al. (2020) J. Cheminf. 12, 52

    该实现会为提取路线数量设置上限，以避免组合爆炸。

    路线会在实例化时直接提取，结果可通过 `routes` 属性访问。

    :param root_node: AND/OR 树根节点
    :param stock: 搜索使用的库存对象
    :param max_routes: 最多提取的路线数
    """

    def __init__(
        self, root_node: TreeNodeMixin, stock: Stock, max_routes: int = 25000
    ) -> None:
        self._traces: List[_AndOrTrace] = []
        self._black_list: List[TreeNodeMixin] = []
        graph = _AndOrTrace(root_node)
        self._samples = {child: 0 for child in root_node.children}
        if root_node.children:
            self._sampling_cutoff = max_routes / len(root_node.children)
        else:
            self._sampling_cutoff = max_routes
        self._partition_search_tree(graph, root_node)
        routes_list = [
            ReactionTreeFromAndOrTrace(trace, stock).tree for trace in self._traces
        ]
        routes_map = {route.hash_key(): route for route in routes_list}
        self.routes = list(routes_map.values())

    def _partition_search_tree(self, graph: _AndOrTrace, node: TreeNodeMixin) -> None:
        # fmt: off
        if self._sampling_cutoff and len(graph) > 1 and self._samples[graph.first_reaction] > self._sampling_cutoff:
            return
        # fmt: on

        if not node.children:
            self._traces.append(graph)

        children_to_search = [
            child for child in node.children if child not in self._black_list
        ]
        if not children_to_search:
            for child in node.children:
                self._black_list.remove(child)
            return

        graph_copy = graph.copy()

        child_node = self._select_child_node(children_to_search)
        graph.add_edge(node, child_node)
        for grandchild in child_node.children:
            graph.add_edge(child_node, grandchild)
        self._black_list.append(child_node)

        leaves = [node for node in graph if not graph[node] and node.children]
        if not leaves:
            self._traces.append(graph)
            self._samples[graph.first_reaction] += 1
        else:
            self._partition_search_tree(graph, leaves[0])

        self._partition_search_tree(graph_copy, node)

    def _select_child_node(self, children: List[TreeNodeMixin]) -> TreeNodeMixin:
        # 这里是该实现与 CompRet 论文原始算法不同的关键点。
        if not self._sampling_cutoff:
            return children[0]

        solved_children = [
            child for child in children if child.prop.get("solved", False)
        ]
        if solved_children:
            return random.choice(solved_children)
        return random.choice(children)


class _AndOrTrace(nx.DiGraph):
    """Helper class for the SplitAndOrTree class."""

    def __init__(self, root: Optional[TreeNodeMixin] = None) -> None:
        super().__init__()
        self.root = root
        self._first_reaction: Optional[TreeNodeMixin] = None
        self._reaction_tree: Optional[Any] = None
        if root:
            self.add_node(root)

    @property
    def first_reaction(self) -> TreeNodeMixin:
        """返回第一步反应；如果不存在则触发断言。"""
        assert self._first_reaction is not None
        return self._first_reaction

    def add_edge(
        self, u_of_edge: TreeNodeMixin, v_of_edge: TreeNodeMixin, **attr: Any
    ) -> None:
        """添加边，并在需要时记录首个反应节点。"""

        if u_of_edge is self.root:
            if self._first_reaction is not None:
                raise ValueError(
                    "Re-defining trace. Trying to set first reaction twice."
                )
            self._first_reaction = v_of_edge
        super().add_edge(u_of_edge, v_of_edge, **attr)

    def copy(self, as_view: bool = False) -> "_AndOrTrace":
        """复制当前跟踪图，并保留根节点与首反应信息。"""

        other = super().copy(as_view)
        assert isinstance(other, _AndOrTrace)
        other.root = self.root
        other._first_reaction = self._first_reaction  # pylint: disable=protected-access
        return other


class ReactionTreeFromAndOrTrace(ReactionTreeLoader):
    """根据 AND/OR 跟踪图创建反应树对象。"""

    def _load(self, andor_trace: nx.DiGraph, stock: Stock) -> None:  # type: ignore
        """
        :param andor_trace: the trace graph
        :param stock: stock object
        """
        self._stock = stock
        self._trace_graph = andor_trace
        self._trace_root = self._find_root()

        self._add_node(
            self._unique_mol(self._trace_root.prop["mol"]),
            depth=0,
            transform=0,
            in_stock=self._trace_root.prop["mol"] in self._stock,
        )
        for node1, node2 in andor_trace.edges():
            if "reaction" in node2.prop and not andor_trace[node2]:
                continue
            rt_node1 = self._make_rt_node(node1)
            rt_node2 = self._make_rt_node(node2)
            self.tree.graph.add_edge(rt_node1, rt_node2)

    def _add_node_with_depth(
        self, node: Union[UniqueMolecule, FixedRetroReaction], base_node: TreeNodeMixin
    ) -> None:
        if node in self.tree.graph:
            return

        depth = nx.shortest_path_length(self._trace_graph, self._trace_root, base_node)
        if isinstance(node, UniqueMolecule):
            self._add_node(
                node, depth=depth, transform=depth // 2, in_stock=node in self._stock
            )
        else:
            self._add_node(node, depth=depth)

    def _find_root(self) -> TreeNodeMixin:
        for node, degree in self._trace_graph.in_degree():  # type: ignore
            if degree == 0:
                return node
        raise ValueError("Could not find root!")

    def _make_rt_node(
        self, node: TreeNodeMixin
    ) -> Union[UniqueMolecule, FixedRetroReaction]:
        if "mol" in node.prop:
            unique_obj = self._unique_mol(node.prop["mol"])
            self._add_node_with_depth(unique_obj, node)
            return unique_obj

        reaction_obj = self._unique_reaction(node.prop["reaction"])
        reaction_obj.reactants = (
            tuple(self._unique_mol(child.prop["mol"]) for child in node.children),
        )
        self._add_node_with_depth(reaction_obj, node)
        return reaction_obj
