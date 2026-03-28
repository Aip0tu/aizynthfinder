"""定义广度优先搜索树节点的模块。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from aizynthfinder.chem import TreeMolecule
from aizynthfinder.chem.serialization import deserialize_action, serialize_action
from aizynthfinder.search.andor_trees import TreeNodeMixin

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.chem.serialization import (
        MoleculeDeserializer,
        MoleculeSerializer,
    )
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import List, Optional, Sequence, Set, StrDict


class MoleculeNode(TreeNodeMixin):
    """
    表示分子的 OR 节点。

    :ivar expandable: 若为 `True`，该节点位于搜索前沿
    :ivar mol: 节点对应的分子
    :ivar in_stock: 若为 `True`，分子已在库存中，无需继续扩展
    :ivar parent: 父节点

    :param mol: 节点要表示的分子
    :param config: 搜索配置
    :param parent: 父节点，可选
    """

    def __init__(
        self,
        mol: TreeMolecule,
        config: Configuration,
        parent: Optional[ReactionNode] = None,
    ) -> None:
        self.mol = mol
        self._config = config
        self.in_stock = mol in config.stock
        self.parent = parent

        self._children: List[ReactionNode] = []
        # 达到最大深度后，该节点不再允许继续扩展。
        self.expandable = self.mol.transform < self._config.search.max_transforms

        if self.in_stock:
            self.expandable = False

    @classmethod
    def create_root(cls, smiles: str, config: Configuration) -> "MoleculeNode":
        """
        Create a root node for a tree using a SMILES.

        :param smiles: the SMILES representation of the root state
        :param config: settings of the tree search algorithm
        :return: the created node
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        return MoleculeNode(mol=mol, config=config)

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        config: Configuration,
        molecules: MoleculeDeserializer,
        parent: Optional[ReactionNode] = None,
    ) -> "MoleculeNode":
        """
        Create a new node from a dictionary, i.e. deserialization

        :param dict_: the serialized node
        :param config: settings of the tree search algorithm
        :param molecules: the deserialized molecules
        :param parent: the parent node
        :return: a deserialized node
        """
        mol = molecules.get_tree_molecules([dict_["mol"]])[0]
        node = MoleculeNode(mol, config, parent)
        node.expandable = dict_["expandable"]
        node.children = [
            ReactionNode.from_dict(child, config, molecules, parent=node)
            for child in dict_["children"]
        ]
        return node

    @property  # type: ignore
    def children(self) -> List[ReactionNode]:  # type: ignore
        """返回反应子节点列表。"""
        return self._children

    @children.setter
    def children(self, value: List[ReactionNode]) -> None:
        self._children = value

    @property
    def prop(self) -> StrDict:
        """返回节点的简要属性视图。"""

        return {"solved": self.in_stock, "mol": self.mol}

    def add_stub(self, reaction: RetroReaction) -> Sequence[MoleculeNode]:
        """
        为当前节点添加一个占位子树。

        :param reaction: 用于创建该子树的反应
        :return: 所有新建分子节点列表
        """
        reactants = reaction.reactants[reaction.index]
        if not reactants:
            return []

        ancestors = self.ancestors()
        for mol in reactants:
            if mol in ancestors:
                return []

        rxn_node = ReactionNode.create_stub(
            reaction=reaction, parent=self, config=self._config
        )
        self._children.append(rxn_node)

        return rxn_node.children

    def ancestors(self) -> Set[TreeMolecule]:
        """
        返回当前节点的祖先分子集合。

        :return: 祖先节点集合
        :rtype: set
        """
        if not self.parent:
            return {self.mol}

        ancestors = self.parent.parent.ancestors()
        ancestors.add(self.mol)
        return ancestors

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        将节点对象序列化为字典。

        :param molecule_store: 分子序列化存储器
        :return: 序列化后的节点字典
        """
        dict_: StrDict = {"expandable": self.expandable}
        dict_["mol"] = molecule_store[self.mol]
        dict_["children"] = [child.serialize(molecule_store) for child in self.children]
        return dict_


class ReactionNode(TreeNodeMixin):
    """
    表示反应的 AND 节点。

    :ivar parent: 父节点
    :ivar reaction: 节点对应的反应

    :param cost: 反应代价
    :param reaction: 节点要表示的反应
    :param parent: 父节点
    """

    def __init__(self, reaction: RetroReaction, parent: MoleculeNode) -> None:
        self.parent = parent
        self.reaction = reaction

        self._children: List[MoleculeNode] = []

    @classmethod
    def create_stub(
        cls,
        reaction: RetroReaction,
        parent: MoleculeNode,
        config: Configuration,
    ) -> ReactionNode:
        """
        创建 `ReactionNode` 及其所有子 `MoleculeNode`。

        :param reaction: 节点要表示的反应
        :param parent: 父节点
        :param config: 搜索树配置
        """
        node = cls(reaction, parent)
        reactants = reaction.reactants[reaction.index]
        node.children = [
            MoleculeNode(mol=mol, config=config, parent=node) for mol in reactants
        ]
        return node

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        config: Configuration,
        molecules: MoleculeDeserializer,
        parent: MoleculeNode,
    ) -> ReactionNode:
        """
        从字典创建新节点，即执行反序列化。

        :param dict_: 序列化后的节点字典
        :param config: 搜索树配置
        :param molecules: 已反序列化的分子对象
        :param parent: 父节点
        :return: 反序列化得到的节点
        """
        reaction = deserialize_action(dict_["reaction"], molecules)
        node = cls(reaction, parent)

        node.children = [
            MoleculeNode.from_dict(child, config, molecules, parent=node)
            for child in dict_["children"]
        ]
        return node

    @property  # type: ignore
    def children(self) -> List[MoleculeNode]:  # type: ignore
        """返回分子子节点列表。"""
        return self._children

    @children.setter
    def children(self, value: List[MoleculeNode]) -> None:
        self._children = value

    @property
    def prop(self) -> StrDict:
        """返回节点的简要属性视图。"""

        return {"solved": False, "reaction": self.reaction}

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        将节点对象序列化为字典。

        :param molecule_store: 分子序列化存储器
        :return: 序列化后的节点字典
        """
        dict_ = {
            "reaction": serialize_action(self.reaction, molecule_store),
            "children": [child.serialize(molecule_store) for child in self.children],
        }
        return dict_
