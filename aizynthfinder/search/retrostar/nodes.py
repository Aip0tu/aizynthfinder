"""定义 Retro* 搜索树节点的模块。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aizynthfinder.chem import TreeMolecule
from aizynthfinder.chem.serialization import deserialize_action, serialize_action
from aizynthfinder.search.andor_trees import TreeNodeMixin
from aizynthfinder.search.retrostar.cost import MoleculeCost

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

    :ivar cost: 合成该分子的代价
    :ivar expandable: 若为 `True`，该节点位于搜索前沿
    :ivar mol: 节点对应的分子
    :ivar in_stock: 若为 `True`，分子已在库存中，无需继续扩展
    :ivar parent: 父节点
    :ivar solved: 若为 `True`，说明分子已在库存中或至少有一个子节点已求解
    :ivar value: 当前 `rn(m|T)` 值

    :param mol: 节点要表示的分子
    :param config: 搜索配置
    :param parent: 父节点，可选
    """

    def __init__(
        self,
        mol: TreeMolecule,
        config: Configuration,
        molecule_cost: MoleculeCost,
        parent: Optional[ReactionNode] = None,
    ) -> None:
        self.mol = mol
        self._config = config
        self.molecule_cost = molecule_cost
        self.cost = self.molecule_cost(mol)
        self.value = self.cost
        self.in_stock = mol in config.stock
        self.parent = parent

        self._children: List[ReactionNode] = []
        self.solved = self.in_stock
        # 达到最大深度后，该节点不再允许继续扩展。
        self.expandable = self.mol.transform < self._config.search.max_transforms

        if self.in_stock:
            self.expandable = False
            self.value = 0

    @classmethod
    def create_root(
        cls, smiles: str, config: Configuration, molecule_cost: MoleculeCost
    ) -> "MoleculeNode":
        """
        使用给定的 SMILES 创建树根节点。

        :param smiles: 根状态对应的 SMILES 表示
        :param config: 树搜索算法配置
        :return: 创建好的根节点
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        return MoleculeNode(mol=mol, config=config, molecule_cost=molecule_cost)

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        config: Configuration,
        molecules: MoleculeDeserializer,
        molecule_cost: MoleculeCost,
        parent: Optional[ReactionNode] = None,
    ) -> "MoleculeNode":
        """
        从字典创建新节点，即执行反序列化。

        :param dict_: 序列化后的节点字典
        :param config: 树搜索算法配置
        :param molecules: 已反序列化的分子对象
        :param parent: 父节点
        :return: 反序列化得到的节点
        """
        mol = molecules.get_tree_molecules([dict_["mol"]])[0]
        node = MoleculeNode(mol, config, molecule_cost, parent)
        node.molecule_cost = molecule_cost
        for attr in ["cost", "expandable", "value"]:
            setattr(node, attr, dict_[attr])
        node.children = [
            ReactionNode.from_dict(
                child, config, molecules, node.molecule_cost, parent=node
            )
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
    def target_value(self) -> float:
        """
        返回 `V_t(m|T)` 值，
        即包含该节点的当前树代价。

        :return: 当前目标值
        """
        if self.parent:
            return self.parent.target_value
        return self.value

    @property
    def prop(self) -> StrDict:
        """返回节点的简要属性视图。"""

        return {"solved": self.solved, "mol": self.mol}

    def add_stub(self, cost: float, reaction: RetroReaction) -> Sequence[MoleculeNode]:
        """
        为当前节点添加一个占位子树。

        :param cost: 反应代价
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
            cost=cost,
            reaction=reaction,
            parent=self,
            config=self._config,
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

    def close(self) -> float:
        """
        在节点展开后更新其数值。

        :return: `V` 值的变化量
        :rtype: float
        """
        self.solved = any(child.solved for child in self.children)
        if self.children:
            new_value = np.min([child.value for child in self.children])
        else:
            new_value = np.inf

        v_delta = new_value - self.value
        self.value = new_value

        self.expandable = False
        return v_delta

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        将节点对象序列化为字典。

        :param molecule_store: 分子序列化存储器
        :return: 序列化后的节点字典
        """
        dict_ = {attr: getattr(self, attr) for attr in ["cost", "expandable", "value"]}
        dict_["mol"] = molecule_store[self.mol]
        dict_["children"] = [child.serialize(molecule_store) for child in self.children]
        return dict_

    def update(self, solved: bool) -> None:
        """
        作为更新算法的一部分刷新当前节点，
        并在需要时继续向父节点传播。

        :param solved: 子节点是否已求解
        """
        new_value = np.min([child.value for child in self.children])
        new_solv = self.solved or solved
        updated = (self.value != new_value) or (self.solved != new_solv)

        v_delta = new_value - self.value
        self.value = new_value
        self.solved = new_solv

        if updated and self.parent:
            self.parent.update(v_delta, from_mol=self.mol)


class ReactionNode(TreeNodeMixin):
    """
    表示反应的 AND 节点。

    :ivar cost: 反应代价
    :ivar parent: 父节点
    :ivar reaction: 节点对应的反应
    :ivar solved: 若为 `True`，说明所有子节点均已求解
    :ivar target_value: 子节点对应的 `V(m|T)` 当前值
    :ivar value: 当前 `rn(r|T)` 值

    :param cost: 反应代价
    :param reaction: 节点要表示的反应
    :param parent: 父节点
    """

    def __init__(
        self, cost: float, reaction: RetroReaction, parent: MoleculeNode
    ) -> None:
        self.parent = parent
        self.cost = cost
        self.reaction = reaction

        self._children: List[MoleculeNode] = []
        self.solved = False
        # rn(R|T)
        self.value = self.cost
        # V(R|T) = 子节点对应的 V(m|T)
        self.target_value = self.parent.target_value - self.parent.value + self.value

    @classmethod
    def create_stub(
        cls,
        cost: float,
        reaction: RetroReaction,
        parent: MoleculeNode,
        config: Configuration,
    ) -> ReactionNode:
        """
        创建 `ReactionNode` 及其所有子 `MoleculeNode`。

        :param cost: 反应代价
        :param reaction: 节点要表示的反应
        :param parent: 父节点
        :param config: 搜索树配置
        """
        node = cls(cost, reaction, parent)
        reactants = reaction.reactants[reaction.index]
        node.children = [
            MoleculeNode(
                mol=mol, config=config, molecule_cost=parent.molecule_cost, parent=node
            )
            for mol in reactants
        ]
        node.solved = all(child.solved for child in node.children)
        # rn(R|T)
        node.value = node.cost + sum(child.value for child in node.children)
        # V(R|T) = V(m|T) for m in children
        node.target_value = node.parent.target_value - node.parent.value + node.value
        return node

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        config: Configuration,
        molecules: MoleculeDeserializer,
        molecule_cost: MoleculeCost,
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
        node = cls(0, reaction, parent)
        for attr in ["cost", "value", "target_value"]:
            setattr(node, attr, dict_[attr])
        node.children = [
            MoleculeNode.from_dict(child, config, molecules, molecule_cost, parent=node)
            for child in dict_["children"]
        ]
        node.solved = all(child.solved for child in node.children)
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

        return {"solved": self.solved, "reaction": self.reaction}

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        将节点对象序列化为字典。

        :param molecule_store: 分子序列化存储器
        :return: 序列化后的节点字典
        """
        dict_ = {
            attr: getattr(self, attr) for attr in ["cost", "value", "target_value"]
        }
        dict_["reaction"] = serialize_action(self.reaction, molecule_store)
        dict_["children"] = [child.serialize(molecule_store) for child in self.children]
        return dict_

    def update(self, value: float, from_mol: Optional[TreeMolecule] = None) -> None:
        """
        作为更新算法的一部分刷新当前节点，
        并继续向父节点传播。

        :param value: `V` 值变化量
        :param from_mol: 当前正在扩展的分子，用于避免重复传播
        """
        self.value += value
        self.target_value += value
        self.solved = all(node.solved for node in self.children)

        if value != 0:
            self._propagate(value, exclude=from_mol)

        self.parent.update(self.solved)

    def _propagate(self, value: float, exclude: Optional[TreeMolecule] = None) -> None:
        if not exclude:
            self.target_value += value

        for child in self.children:
            if exclude is None or child.mol is not exclude:
                for grandchild in child.children:
                    grandchild._propagate(value)  # pylint: disable=protected-access
