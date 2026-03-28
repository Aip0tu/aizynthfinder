"""定义路线评分相关基础类的模块。"""

from __future__ import annotations

import abc
import functools
from collections.abc import Sequence as SequenceAbc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rxnutils.routes.readers import read_aizynthfinder_dict

from aizynthfinder.reactiontree import ReactionTree
from aizynthfinder.search.mcts import MctsNode
from aizynthfinder.utils.exceptions import ScorerException

if TYPE_CHECKING:
    from rxnutils.routes.base import SynthesisRoute

    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import (
        Optional,
        Sequence,
        StrDict,
        Tuple,
        TypeVar,
        Union,
    )

    _Scoreable = TypeVar("_Scoreable", MctsNode, ReactionTree)
    _Scoreables = Sequence[_Scoreable]
    _ScorerItemType = Union[_Scoreables, _Scoreable]


@dataclass
class SquashScaler:
    """
    通过可调参数对 sigmoid 进行变形后的压缩函数。

    :param slope: 中点附近的斜率
    :param xoffset: 曲线在 x 轴上的中点偏移
    :param yoffset: 曲线在 y 轴上的整体偏移
    """

    slope: float
    xoffset: float
    yoffset: float

    def __call__(self, val: float) -> float:
        return 1 / (1 + np.exp(self.slope * -(val - self.xoffset))) - self.yoffset


@dataclass
class MinMaxScaler:
    """
    将值归一化到 0 到 1 之间的缩放函数。

    `reverse` 控制缩放方向；对于需要最小化的奖励，应设为 `True`。
    `scale_factor` 可在分数过小或过大时进一步调整结果。

    :param val: 待缩放的值
    :param min_val: `val` 的理论最小值
    :param max_val: `val` 的理论最大值
    :param scale_factor: 归一化后额外施加的缩放因子
    """

    min_val: float
    max_val: float
    reverse: bool
    scale_factor: float = 1

    def __call__(self, val: float) -> float:
        val = np.clip(val, self.min_val, self.max_val)
        if self.reverse:
            numerator = self.max_val - val
        else:
            numerator = val - self.min_val
        return (numerator / (self.max_val - self.min_val)) * self.scale_factor


@dataclass
class PowerScaler:
    """
    以底数幂函数形式进行缩放。

    :param val: 待缩放的值
    :param base_coefficient: 缩放底数，`base_coefficient < 1` 时会反转缩放方向
    """

    base_coefficient: float = 0.98

    def __call__(self, val: float) -> float:
        return self.base_coefficient**val


_SCALERS = {"squash": SquashScaler, "min_max": MinMaxScaler, "power": PowerScaler}


class Scorer(abc.ABC):
    """
    对 MCTS 风格节点或反应树进行评分的抽象基类。

    评分时，直接把 `Node` 或 `ReactionTree` 作为唯一参数传给评分器实例即可。

    .. code-block::

        scorer = MyScorer()
        score = scorer(node1)

    也可以一次传入由这些对象组成的列表。

    .. code-block::

        scorer = MyScorer()
        scores = scorer([node1, node2])

    :param config: 树搜索配置
    :param scaler_params: 缩放器参数设置
    """

    scorer_name = "base"

    def __init__(
        self,
        config: Optional[Configuration] = None,
        scaler_params: Optional[StrDict] = None,
    ) -> None:
        self._config = config
        self._reverse_order: bool = True
        self._scaler = None
        self._scaler_name = ""
        if scaler_params:
            self._scaler_name = scaler_params["name"]
            del scaler_params["name"]
            if scaler_params:
                self._scaler = _SCALERS[self._scaler_name](**scaler_params)
            else:
                # for parameterless function
                self._scaler = _SCALERS[self._scaler_name]()

    def __call__(self, item: _ScorerItemType) -> Union[float, Sequence[float]]:
        if isinstance(item, SequenceAbc):
            return self._score_many(item)
        if isinstance(item, (MctsNode, ReactionTree)):
            return self._score_just_one(item)  # type: ignore
        raise ScorerException(
            f"Unable to score item from class {item.__class__.__name__}"
        )

    def __repr__(self) -> str:
        repr_name = self.scorer_name
        if self._scaler_name:
            repr_name += f"-{self._scaler_name}"
        return repr_name

    def sort(
        self, items: _Scoreables
    ) -> Tuple[_Scoreables, Sequence[float], Sequence[int]]:
        """
        按分数从高到低对节点或反应树排序。

        :param items: 待排序的条目
        :return: 排序后的条目及其分数
        """
        scores = self._score_many(items)
        assert isinstance(scores, SequenceAbc)
        sortidx = sorted(
            range(len(scores)), key=scores.__getitem__, reverse=self._reverse_order
        )
        scores = [scores[idx] for idx in sortidx]
        sorted_items = [items[idx] for idx in sortidx]
        return sorted_items, scores, sortidx

    def _score_just_one(self, item: _Scoreable) -> float:
        if isinstance(item, MctsNode):
            node_score = self._score_node(item)
            if self._scaler:
                node_score = self._scaler(node_score)
            return node_score
        if isinstance(item, ReactionTree):
            tree_score = self._score_reaction_tree(item)
            if self._scaler:
                tree_score = self._scaler(tree_score)
            return tree_score
        raise ScorerException(
            f"Unable to score item from class {item.__class__.__name__}"
        )

    def _score_many(self, items: _Scoreables) -> Sequence[float]:
        return [self._score_just_one(item) for item in items]

    @abc.abstractmethod
    def _score_node(self, node: MctsNode) -> float:
        pass

    @abc.abstractmethod
    def _score_reaction_tree(self, tree: ReactionTree) -> float:
        pass


@functools.lru_cache
def make_rxnutils_route(tree: ReactionTree) -> SynthesisRoute:
    """将 `ReactionTree` 转换为 `rxnutils` 可识别的路线对象。"""

    return read_aizynthfinder_dict(tree.to_dict())
