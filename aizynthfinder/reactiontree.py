"""实现反应树（路线）及其构建工厂类的模块。"""

from __future__ import annotations

import abc
import hashlib
import json
import operator
from typing import TYPE_CHECKING

import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from rxnutils.routes.comparison import simple_route_similarity
from rxnutils.routes.image import RouteImageFactory
from rxnutils.routes.readers import read_aizynthfinder_dict

from aizynthfinder.chem import (
    FixedRetroReaction,
    Molecule,
    UniqueMolecule,
    none_molecule,
)

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.utils.type_utils import (
        Any,
        Dict,
        FrameColors,
        Iterable,
        List,
        Optional,
        PilImage,
        StrDict,
        Union,
    )


class ReactionTree:
    """
    封装单条路线的二分反应树。

    树中的节点由 `FixedRetroReaction` 或 `UniqueMolecule` 组成。
    反应树在实例化时完成初始化，后续通常不再被修改。

    :ivar graph: 二分图结构
    :ivar is_solved: 所有叶子节点是否都在库存中
    :ivar root: 树的根节点
    :ivar created_at_iteration: 创建该反应树时对应的迭代次数
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.root = none_molecule()
        self.is_solved: bool = False
        self.created_at_iteration: Optional[int] = None

    @classmethod
    def from_dict(cls, tree_dict: StrDict) -> "ReactionTree":
        """
        通过解析字典创建新的 `ReactionTree`。

        该方法理论上与 `to_dict` 相对应，
        但由于该格式会丢失部分信息，返回对象并不是完整拷贝；
        其库存信息只会包含字典中标记为 `in_stock` 的分子。

        不过，返回结果通常已足够用于生成路线图像等场景。

        :param tree_dict: 反应树的字典表示
        :returns: 构建得到的反应树
        """
        return ReactionTreeFromDict(tree_dict).tree

    @property
    def metadata(self) -> StrDict:
        """返回包含路线元数据的字典。"""
        return {
            "created_at_iteration": self.created_at_iteration,
            "is_solved": self.is_solved,
        }

    def child_reactions(self, reaction: FixedRetroReaction) -> List[FixedRetroReaction]:
        """
        返回树中某个反应节点的子反应节点。

        :param reaction: 查询节点（反应）
        :return: 子反应节点列表
        """
        child_molecule_nodes = self.graph.successors(reaction)
        reaction_nodes = []
        for molecule_node in child_molecule_nodes:
            reaction_nodes.extend(list(self.graph.successors(molecule_node)))
        return reaction_nodes

    def depth(self, node: Union[UniqueMolecule, FixedRetroReaction]) -> int:
        """
        返回节点在路线中的深度。

        :param node: 查询节点
        :return: 节点深度
        """
        return self.graph.nodes[node].get("depth", -1)

    def distance_to(self, other: "ReactionTree") -> float:
        """
        计算当前反应树与另一棵反应树之间的距离。

        距离基于路线相似度进行简单换算。

        :param other: 要比较的另一棵反应树
        :return: 两条路线之间的距离
        """
        route1 = read_aizynthfinder_dict(self.to_dict())
        route2 = read_aizynthfinder_dict(other.to_dict())
        return 1.0 - float(simple_route_similarity([route1, route2])[0, 1])

    def hash_key(self) -> str:
        """
        使用 `sha224` 递归计算整棵树的哈希值。

        :return: 哈希键
        """
        return self._hash_func(self.root)

    def in_stock(self, node: Union[UniqueMolecule, FixedRetroReaction]) -> bool:
        """
        返回路线中的节点是否在库存中。

        注意，这个属性在创建时写入，之后不会自动更新。

        :param node: 查询节点
        :return: 分子是否在库存中
        """
        return self.graph.nodes[node].get("in_stock", False)

    def is_branched(self) -> bool:
        """
        判断路线是否为分支结构。

        具体来说，会检查最大深度是否不等于反应步骤数。
        """
        nsteps = len(list(self.reactions()))
        max_depth = max(self.depth(leaf) for leaf in self.leafs())
        return nsteps != max_depth // 2

    def leafs(self) -> Iterable[UniqueMolecule]:
        """
        生成反应树中没有后继的分子节点，
        即尚未继续被拆解的分子。

        :yield: 下一个叶子分子节点
        """
        for node in self.graph:
            if isinstance(node, UniqueMolecule) and not self.graph[node]:
                yield node

    def molecules(self) -> Iterable[UniqueMolecule]:
        """
        生成反应树中的所有分子节点。

        :yield: 下一个分子节点
        """
        for node in self.graph:
            if isinstance(node, UniqueMolecule):
                yield node

    def parent_molecule(self, mol: UniqueMolecule) -> UniqueMolecule:
        """返回反应树中某个分子节点的父分子节点。

        :param mol: 查询节点（分子）
        :return: 父分子节点
        """
        if mol is self.root:
            raise ValueError("Root molecule does not have any parent node.")

        parent_reaction = list(self.graph.predecessors(mol))[0]
        parent_molecule = list(self.graph.predecessors(parent_reaction))[0]
        return parent_molecule

    def reactions(self) -> Iterable[FixedRetroReaction]:
        """
        生成反应树中的所有反应节点。

        :yield: 下一个反应节点
        """
        for node in self.graph:
            if not isinstance(node, Molecule):
                yield node

    def subtrees(self) -> Iterable[ReactionTree]:
        """
        生成当前反应树的所有子树。

        子树指的是从某个仍有子节点的分子节点开始的反应树。

        :yield: 下一个子树
        """

        def create_subtree(root_node):
            subtree = ReactionTree()
            subtree.root = root_node
            subtree.graph = dfs_tree(self.graph, root_node)
            for node in subtree.graph:
                prop = dict(self.graph.nodes[node])
                prop["depth"] -= self.graph.nodes[root_node].get("depth", 0)
                if "transform" in prop:
                    prop["transform"] -= self.graph.nodes[root_node].get("transform", 0)
                subtree.graph.nodes[node].update(prop)
            subtree.is_solved = all(subtree.in_stock(node) for node in subtree.leafs())
            return subtree

        for node in self.molecules():
            if node is not self.root and self.graph[node]:
                yield create_subtree(node)

    def to_dict(self, include_metadata=False) -> StrDict:
        """
        按预定义格式将反应树转换为字典。

        :param include_metadata: 若为 `True`，则包含元数据
        :return: 反应树字典
        """
        return self._build_dict(self.root, include_metadata=include_metadata)

    def to_image(
        self,
        in_stock_colors: Optional[FrameColors] = None,
        show_all: bool = True,
    ) -> PilImage:
        """
        返回路线的图像表示。

        :param in_stock_colors: 分子边框颜色，默认为 `{True: "green", False: "orange"}`
        :param show_all: 若为 `True`，也显示被标记为隐藏的节点
        :return: 路线图像
        """
        factory = RouteImageFactory(
            self.to_dict(), in_stock_colors=in_stock_colors, show_all=show_all
        )
        return factory.image

    def to_json(self, include_metadata=False) -> str:
        """
        按预定义格式将反应树转换为 JSON 字符串。

        :return: JSON 字符串形式的反应树
        """
        return json.dumps(
            self.to_dict(include_metadata=include_metadata), sort_keys=False, indent=2
        )

    def _build_dict(
        self,
        node: Union[UniqueMolecule, FixedRetroReaction],
        dict_: Optional[StrDict] = None,
        include_metadata=False,
    ) -> StrDict:
        if dict_ is None:
            dict_ = {}

        if node is self.root and include_metadata:
            dict_["route_metadata"] = self.metadata

        dict_["type"] = "mol" if isinstance(node, Molecule) else "reaction"
        dict_["hide"] = self.graph.nodes[node].get("hide", False)
        dict_["smiles"] = node.smiles
        if isinstance(node, UniqueMolecule):
            dict_["is_chemical"] = True
            dict_["in_stock"] = self.in_stock(node)
        elif isinstance(node, FixedRetroReaction):
            dict_["is_reaction"] = True
            dict_["metadata"] = dict(node.metadata)
        else:
            raise ValueError(
                f"This is an invalid reaction tree. Unknown node type {type(node)}"
            )

        dict_["children"] = []

        children = list(self.graph.successors(node))
        if isinstance(node, FixedRetroReaction):
            children.sort(key=operator.attrgetter("weight"))
        for child in children:
            child_dict = self._build_dict(child)
            dict_["children"].append(child_dict)

        if not dict_["children"]:
            del dict_["children"]
        return dict_

    def _hash_func(self, node: Union[FixedRetroReaction, UniqueMolecule]) -> str:
        if isinstance(node, UniqueMolecule):
            hash_ = hashlib.sha224(node.inchi_key.encode())
        else:
            hash_ = hashlib.sha224(node.hash_key().encode())
        child_hashes = sorted(
            self._hash_func(child) for child in self.graph.successors(node)
        )
        for child_hash in child_hashes:
            hash_.update(child_hash.encode())
        return hash_.hexdigest()


class ReactionTreeLoader(abc.ABC):
    """
    用于创建反应树对象的基类。

    该类负责在图生成后统一设置节点属性，并提供若干辅助方法。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._unique_mols: Dict[int, UniqueMolecule] = {}
        self._unique_reactions: Dict[int, FixedRetroReaction] = {}
        self.tree = ReactionTree()
        self._load(*args, **kwargs)

        self.tree.is_solved = all(
            self.tree.in_stock(node) for node in self.tree.leafs()
        )

    def _add_node(
        self,
        node: Union[UniqueMolecule, FixedRetroReaction],
        depth: int = 0,
        transform: int = 0,
        in_stock: bool = False,
        hide: bool = False,
    ) -> None:
        attributes = {
            "hide": hide,
            "depth": depth,
        }
        if isinstance(node, Molecule):
            attributes.update({"transform": transform, "in_stock": in_stock})
            if not self.tree.root:
                self.tree.root = node
        self.tree.graph.add_node(node, **attributes)

    @abc.abstractmethod
    def _load(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _unique_mol(self, molecule: Molecule) -> UniqueMolecule:
        id_ = id(molecule)
        if id_ not in self._unique_mols:
            self._unique_mols[id_] = molecule.make_unique()
        return self._unique_mols[id_]

    def _unique_reaction(self, reaction: RetroReaction) -> FixedRetroReaction:
        id_ = id(reaction)
        if id_ not in self._unique_reactions:
            metadata = dict(reaction.metadata)
            if ":" in reaction.mapped_reaction_smiles():
                metadata["mapped_reaction_smiles"] = reaction.mapped_reaction_smiles()
            self._unique_reactions[id_] = FixedRetroReaction(
                self._unique_mol(reaction.mol),
                smiles=reaction.smiles,
                metadata=metadata,
            )
        return self._unique_reactions[id_]


class ReactionTreeFromDict(ReactionTreeLoader):
    """从字典创建反应树对象。"""

    def _load(self, tree_dict: StrDict) -> None:  # type: ignore
        if tree_dict.get("route_metadata"):
            self.tree.created_at_iteration = tree_dict["route_metadata"].get(
                "created_at_iteration"
            )
        self._parse_tree_dict(tree_dict)

    def _parse_tree_dict(self, tree_dict: StrDict, ncalls: int = 0) -> UniqueMolecule:
        product_node = UniqueMolecule(smiles=tree_dict["smiles"])
        self._add_node(
            product_node,
            depth=2 * ncalls,
            transform=ncalls,
            hide=tree_dict.get("hide", False),
            in_stock=tree_dict["in_stock"],
        )

        rxn_tree_dict = tree_dict.get("children", [])
        if not rxn_tree_dict:
            return product_node

        rxn_tree_dict = rxn_tree_dict[0]
        reaction_node = FixedRetroReaction(
            product_node,
            smiles=rxn_tree_dict["smiles"],
            metadata=rxn_tree_dict.get("metadata", {}),
        )
        self._add_node(
            reaction_node, depth=2 * ncalls + 1, hide=rxn_tree_dict.get("hide", False)
        )
        self.tree.graph.add_edge(product_node, reaction_node)

        reactant_nodes = []
        for reactant_tree in rxn_tree_dict.get("children", []):
            reactant_node = self._parse_tree_dict(reactant_tree, ncalls + 1)
            self.tree.graph.add_edge(reaction_node, reactant_node)
            reactant_nodes.append(reactant_node)
        reaction_node.reactants = (tuple(reactant_nodes),)

        return product_node


class ReactionTreeFromExpansion(ReactionTreeLoader):
    """
    根据单个反应创建 `ReactionTree`。

    这主要是为扩展器接口提供的便捷构造方式。
    """

    def _load(self, reaction: RetroReaction) -> None:  # type: ignore
        root = self._unique_mol(reaction.mol)
        self._add_node(root)

        rxn = self._unique_reaction(reaction)
        if hasattr(reaction, "smarts"):
            rxn.metadata["smarts"] = reaction.smarts  # type: ignore
        self._add_node(rxn)
        self.tree.graph.add_edge(root, rxn)

        reactant_nodes = []
        for reactant in reaction.reactants[0]:
            reactant_node = self._unique_mol(reactant)
            reactant_nodes.append(reactant_node)
            self._add_node(reactant_node)
            self.tree.graph.add_edge(rxn, reactant_node)
        rxn.reactants = (tuple(reactant_nodes),)
