"""Retro* 分子代价模型定义。"""
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

from aizynthfinder.search.retrostar.cost import __name__ as retrostar_cost_module
from aizynthfinder.utils.loading import load_dynamic_class

if TYPE_CHECKING:
    from aizynthfinder.chem import Molecule
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import Any, List, Tuple


class MoleculeCost:
    """
    负责计算分子代价的封装类。

    要使用的代价模型从配置中读取。
    如果未设置 `molecule_cost`，则默认使用 `ZeroMoleculeCost`。
    `molecule_cost` 可以在 `search` 下按如下结构配置：
    'algorithm': 'retrostar'
    'algorithm_config': {
        'molecule_cost': {
            'cost': name of the search cost class or custom_package.custom_model.CustomClass,
            other settings or params
        }
    }

    实例化后，可以直接把分子对象传给该类来计算代价。

    .. code-block::

        calculator = MyCost(config)
        cost = calculator.calculate(molecule)

    :param config: 树搜索配置
    """

    def __init__(self, config: Configuration) -> None:
        self._config = config
        if "molecule_cost" not in self._config.search.algorithm_config:
            self._config.search.algorithm_config["molecule_cost"] = {
                "cost": "ZeroMoleculeCost"
            }
        kwargs = self._config.search.algorithm_config["molecule_cost"].copy()

        cls = load_dynamic_class(kwargs["cost"], retrostar_cost_module)
        del kwargs["cost"]

        self.molecule_cost = cls(**kwargs) if kwargs else cls()

    def __call__(self, mol: Molecule) -> float:
        return self.molecule_cost.calculate(mol)


class RetroStarCost:
    """
    原始 Retro* 分子代价模型的封装。

    这是对原始 PyTorch 模型的 NumPy 实现。

    评分对象是 `Molecule` 实例。

    .. code-block::

        mol = Molecule(smiles="CCC")
        scorer = RetroStarCost()
        score = scorer.calculate(mol)

    创建评分器时提供的模型应是一个 pickled 元组。
    元组第一个元素是各层权重列表，
    第二个元素是各层偏置列表。

    :param model_path: 模型权重与偏置文件路径
    :param fingerprint_length: 指纹位数
    :param fingerprint_radius: 指纹半径
    :param dropout_rate: dropout 比例
    """

    _required_kwargs = ["model_path"]

    def __init__(self, **kwargs: Any) -> None:
        model_path = kwargs["model_path"]
        self.fingerprint_length: int = int(kwargs.get("fingerprint_length", 2048))
        self.fingerprint_radius: int = int(kwargs.get("fingerprint_radius", 2))
        self.dropout_rate: float = float(kwargs.get("dropout_rate", 0.1))

        self._dropout_prob = 1.0 - self.dropout_rate
        self._weights, self._biases = self._load_model(model_path)

    def __repr__(self) -> str:
        return "retrostar"

    def calculate(self, mol: Molecule) -> float:
        """计算指定分子的 Retro* 代价值。"""

        # pylint: disable=invalid-name
        mol.sanitize()
        vec = mol.fingerprint(
            radius=self.fingerprint_radius, nbits=self.fingerprint_length
        )
        for W, b in zip(self._weights[:-1], self._biases[:-1]):
            vec = np.matmul(vec, W) + b
            vec *= vec > 0  # ReLU
            # Dropout
            vec *= np.random.binomial(1, self._dropout_prob, size=vec.shape) / (
                self._dropout_prob
            )
        vec = np.matmul(vec, self._weights[-1]) + self._biases[-1]
        return float(np.log(1 + np.exp(vec)))

    @staticmethod
    def _load_model(model_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        with open(model_path, "rb") as fileobj:
            weights, biases = pickle.load(fileobj)

        return (
            [np.asarray(item) for item in weights],
            [np.asarray(item) for item in biases],
        )


class ZeroMoleculeCost:
    """零代价模型封装。"""

    def __repr__(self) -> str:
        return "zero"

    def calculate(self, _mol: Molecule) -> float:  # pytest: disable=unused-argument
        """始终返回零代价。"""

        return 0.0
