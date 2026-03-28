"""识别关注键断裂情况的工具模块。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Sequence, Tuple
    from aizynthfinder.chem.mol import TreeMolecule
    from aizynthfinder.chem.reaction import RetroReaction


class BrokenBonds:
    """
    跟踪目标分子中关注键断裂情况的工具类。

    :param focussed_bonds: 关注键对列表。
        每个键对都由长度为 2 的元组表示，并且这些键应存在于目标分子的原子键中。
    """

    def __init__(self, focussed_bonds: Sequence[Sequence[int]]) -> None:
        self.focussed_bonds = sort_bonds(focussed_bonds)
        self.filtered_focussed_bonds: List[Tuple[int, int]] = []

    def __call__(self, reaction: RetroReaction) -> List[Tuple[int, int]]:
        """
        返回在反应物中被断开的关注键列表。

        :param reaction: 逆合成反应
        :return: 目标分子对应反应物中所有已断开的关注键列表
        """
        self.filtered_focussed_bonds = self._get_filtered_focussed_bonds(reaction.mol)
        if not self.filtered_focussed_bonds:
            return []

        molecule_bonds = []
        for reactant in reaction.reactants[reaction.index]:
            molecule_bonds += reactant.mapped_atom_bonds

        broken_focussed_bonds = self._get_broken_frozen_bonds(
            sort_bonds(molecule_bonds)
        )
        return broken_focussed_bonds

    def _get_broken_frozen_bonds(
        self,
        molecule_bonds: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        broken_focussed_bonds = list(
            set(self.filtered_focussed_bonds) - set(molecule_bonds)
        )
        return broken_focussed_bonds

    def _get_filtered_focussed_bonds(
        self, molecule: TreeMolecule
    ) -> List[Tuple[int, int]]:
        molecule_bonds = molecule.mapped_atom_bonds
        atom_maps = [atom_map for bonds in molecule_bonds for atom_map in bonds]

        filtered_focussed_bonds = []
        for idx1, idx2 in self.focussed_bonds:
            if idx1 in atom_maps and idx2 in atom_maps:
                filtered_focussed_bonds.append((idx1, idx2))
        return filtered_focussed_bonds


def sort_bonds(bonds: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """对键对中的原子索引做排序并规范化为元组列表。"""

    return [tuple(sorted(bond)) for bond in bonds]  # type: ignore
