"""定义 DFPN 搜索中各类树节点的模块。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aizynthfinder.chem import TreeMolecule
from aizynthfinder.search.andor_trees import TreeNodeMixin

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.search.dfpn import SearchTree
    from aizynthfinder.utils.type_utils import List, Optional, Sequence, Set, StrDict

BIG_INT = int(1e10)


class _SuperNode(TreeNodeMixin):
    """DFPN 节点的公共基类，负责维护证明数与反证数。"""

    def __init__(self) -> None:
        # pylint: disable=invalid-name
        self.pn = 1  # 证明数
        self.dn = 1  # 反证数
        self.pn_threshold = BIG_INT
        self.dn_threshold = BIG_INT
        self._children: List["_SuperNode"] = []
        self.expandable = True

    @property  # type: ignore
    def children(self) -> List[ReactionNode]:  # type: ignore
        """返回反应子节点列表。"""
        return self._children  # type: ignore

    @property
    def closed(self) -> bool:
        """返回节点是否已经被证明或反证。"""
        return self.proven or self.disproven

    @property
    def proven(self) -> bool:
        """返回节点是否已被证明。"""
        return self.pn == 0

    @property
    def disproven(self) -> bool:
        """返回节点是否已被反证。"""
        return self.dn == 0

    def explorable(self) -> bool:
        """返回该节点是否仍可被搜索算法继续探索。"""
        return not (
            self.closed or self.pn > self.pn_threshold or self.dn > self.dn_threshold
        )

    def reset(self) -> None:
        """重置当前节点及其子节点的阈值。"""
        if self.closed or self.expandable:
            return
        for child in self._children:
            child.reset()
        self.update()
        self.pn_threshold = BIG_INT
        self.dn_threshold = BIG_INT

    def update(self) -> None:
        """更新证明数与反证数。"""
        raise NotImplementedError("Implement a child class")

    def _set_disproven(self) -> None:
        self.pn = BIG_INT
        self.dn = 0

    def _set_proven(self) -> None:
        self.pn = 0
        self.dn = BIG_INT


class MoleculeNode(_SuperNode):
    """
    表示分子的 OR 节点。

    :ivar expandable: 若为 `True`，说明该节点位于搜索前沿
    :ivar mol: 节点对应的分子
    :ivar in_stock: 若为 `True`，说明分子已在库存中，无需继续扩展
    :ivar parent: 父节点
    :ivar pn: 证明数
    :ivar dn: 反证数
    :ivar pn_threshold: 证明数阈值
    :ivar dn_threshold: 反证数阈值

    :param mol: 节点要表示的分子
    :param config: 搜索配置
    :param parent: 父节点，可选
    """

    def __init__(
        self,
        mol: TreeMolecule,
        config: Configuration,
        owner: SearchTree,
        parent: Optional[ReactionNode] = None,
    ) -> None:
        super().__init__()

        self.mol = mol
        self._config = config
        self.in_stock = mol in config.stock
        self.parent = parent
        self._edge_costs: List[int] = []
        self.tree = owner

        # 达到最大深度后，该节点不再允许继续扩展。
        self.expandable = self.mol.transform < self._config.search.max_transforms

        if self.in_stock:
            self.expandable = False
            self._set_proven()
        elif not self.expandable:
            self._set_disproven()

    @classmethod
    def create_root(
        cls, smiles: str, config: Configuration, owner: SearchTree
    ) -> "MoleculeNode":
        """
        使用给定的 SMILES 创建树根节点。

        :param smiles: 根状态对应的 SMILES 表示
        :param config: 树搜索算法配置
        :return: 创建好的根节点
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        return MoleculeNode(mol=mol, config=config, owner=owner)

    @property
    def prop(self) -> StrDict:
        """返回节点的简要属性视图。"""

        return {"solved": self.proven, "mol": self.mol}

    def expand(self) -> None:
        """利用扩展策略展开当前分子节点。"""
        self.expandable = False
        reactions, priors = self._config.expansion_policy([self.mol])
        self.tree.profiling["expansion_calls"] += 1

        if not reactions:
            self._set_disproven()
            return

        costs = -np.log(np.clip(priors, 1e-3, 1.0))
        reaction_costs = []
        reactions_to_expand = []
        for reaction, cost in zip(reactions, costs):
            try:
                _ = reaction.reactants
                self.tree.profiling["reactants_generations"] += 1
            except:  # pylint: disable=bare-except
                continue
            if not reaction.reactants:
                continue
            for idx, _ in enumerate(reaction.reactants):
                rxn_copy = reaction.copy(idx)
                reactions_to_expand.append(rxn_copy)
                reaction_costs.append(cost)

        for cost, rxn in zip(reaction_costs, reactions_to_expand):
            self._add_child(rxn, cost)

        if not self._children:
            self._set_disproven()

    def promising_child(self) -> Optional[ReactionNode]:
        """
        找出并返回当前最值得继续探索的子节点。

        同时会更新该子节点的阈值。
        """
        min_indices = np.argsort(
            [
                edge_cost + child.pn if not child.closed else BIG_INT
                for edge_cost, child in zip(self._edge_costs, self._children)
            ]
        )
        best_child = self._children[min_indices[0]]
        if len(self._children) > 1 and not self._children[min_indices[1]].closed:
            s2_pn = self._children[min_indices[1]].pn
        else:
            s2_pn = BIG_INT

        best_child.pn_threshold = (
            min(self.pn_threshold, s2_pn + 2) - self._edge_costs[min_indices[0]]
        )
        best_child.dn_threshold = self.dn_threshold - self.dn + best_child.dn
        return best_child

    def update(self) -> None:
        """更新证明数与反证数。"""
        func = all if self.parent is None else any
        if func(child.proven for child in self._children):
            self._set_proven()
            return
        if all(child.disproven for child in self._children):
            self._set_disproven()
            return

        child_dns = [child.dn for child in self._children if not child.closed]
        if not child_dns:
            self._set_proven()
            return

        self.dn = sum(child_dns)
        if self.dn >= BIG_INT:
            self.pn = 0
        else:
            self.pn = min(
                edge_cost + child.pn
                for edge_cost, child in zip(self._edge_costs, self._children)
                if not child.closed
            )
        return

    def _add_child(self, reaction: RetroReaction, _: float) -> None:
        reactants = reaction.reactants[reaction.index]
        if not reactants:
            return

        ancestors = self._ancestors()
        for mol in reactants:
            if mol in ancestors:
                return

        rxn_node = ReactionNode(
            reaction=reaction, config=self._config, owner=self.tree, parent=self
        )
        self._children.append(rxn_node)
        self._edge_costs.append(1)

    def _ancestors(self) -> Set[TreeMolecule]:
        if not self.parent:
            return {self.mol}

        # pylint: disable=protected-access
        ancestors = self.parent.parent._ancestors()
        ancestors.add(self.mol)
        return ancestors


class ReactionNode(_SuperNode):
    """
    表示反应的 AND 节点。

    :ivar parent: 父节点
    :ivar reaction: 节点对应的反应
    :ivar pn: 证明数
    :ivar dn: 反证数
    :ivar pn_threshold: 证明数阈值
    :ivar dn_threshold: 反证数阈值
    :ivar expandable: 节点是否可继续扩展

    :param reaction: 节点要表示的反应
    :param config: 搜索配置
    :param parent: 父节点
    """

    def __init__(
        self,
        reaction: RetroReaction,
        config: Configuration,
        owner: SearchTree,
        parent: MoleculeNode,
    ) -> None:
        super().__init__()
        self._config = config
        self.parent = parent
        self.reaction = reaction
        self.tree = owner

    @property  # type: ignore
    def children(self) -> List[MoleculeNode]:  # type: ignore
        """返回分子子节点列表。"""
        return self._children  # type: ignore

    @property
    def prop(self) -> StrDict:
        """返回节点的简要属性视图。"""

        return {"solved": self.proven, "reaction": self.reaction}

    @property
    def proven(self) -> bool:
        """返回节点是否已被证明。"""
        if self.expandable:
            return False
        if self.pn == 0:
            return True
        return all(child.proven for child in self._children)

    @property
    def disproven(self) -> bool:
        """返回节点是否已被反证。"""
        if self.expandable:
            return False
        if self.dn == 0:
            return True
        return any(child.disproven for child in self._children)

    def expand(self) -> None:
        """为每个反应物创建节点，从而展开当前反应节点。"""
        self.expandable = False
        reactants = self.reaction.reactants[self.reaction.index]
        self._children = [
            MoleculeNode(mol=mol, config=self._config, owner=self.tree, parent=self)
            for mol in reactants
        ]

    def promising_child(self) -> Optional[MoleculeNode]:
        """
        找出并返回当前最值得继续探索的子节点。

        同时会更新该子节点的阈值。
        """
        min_indices = np.argsort(
            [child.dn if not child.closed else BIG_INT for child in self._children]
        )

        best_child = self._children[min_indices[0]]
        if len(self._children) > 1 and not self._children[min_indices[1]].closed:
            s2_dn = self._children[min_indices[1]].dn
        else:
            s2_dn = BIG_INT

        best_child.pn_threshold = self.pn_threshold - self.pn + best_child.pn
        best_child.dn_threshold = min(self.dn_threshold, s2_dn + 1)
        return best_child

    def update(self) -> None:
        """更新证明数与反证数。"""
        if all(child.proven for child in self._children):
            self._set_proven()
            return
        if any(child.disproven for child in self._children):
            self._set_disproven()
            return

        self.pn = sum(child.pn for child in self._children if not child.closed)
        self.dn = min(child.dn for child in self._children if not child.closed)
