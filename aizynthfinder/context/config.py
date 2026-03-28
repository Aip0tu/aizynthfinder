"""封装树搜索配置的模块。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

from aizynthfinder.context.policy import ExpansionPolicy, FilterPolicy
from aizynthfinder.context.scoring import ScorerCollection
from aizynthfinder.context.stock import Stock
from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.utils.type_utils import Any, Dict, List, Optional, StrDict, Union


@dataclass
class _PostprocessingConfiguration:
    """定义路线后处理阶段的配置项。"""

    min_routes: int = 5
    max_routes: int = 25
    all_routes: bool = False
    route_distance_model: Optional[str] = None
    route_scorers: List[str] = field(default_factory=lambda: [])
    scorer_weights: Optional[List[float]] = field(default_factory=lambda: None)


@dataclass
class _SearchConfiguration:
    """定义搜索算法及其运行时参数。"""

    algorithm: str = "mcts"
    algorithm_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 1.4,
            "default_prior": 0.5,
            "use_prior": True,
            "prune_cycles_in_search": True,
            "search_rewards": ["state score"],
            "immediate_instantiation": (),
            "mcts_grouping": None,
            "search_rewards_weights": [],
        }
    )
    max_transforms: int = 6
    iteration_limit: int = 100
    time_limit: int = 120
    return_first: bool = False
    exclude_target_from_stock: bool = True
    break_bonds: List[List[int]] = field(default_factory=list)
    freeze_bonds: List[List[int]] = field(default_factory=list)
    break_bonds_operator: str = "and"


@dataclass
class Configuration:
    """
    封装树搜索所需的全部配置，包括策略、库存、评分器及各类参数。
    """

    search: _SearchConfiguration = field(default_factory=_SearchConfiguration)
    post_processing: _PostprocessingConfiguration = field(
        default_factory=_PostprocessingConfiguration
    )
    stock: Stock = field(init=False)
    expansion_policy: ExpansionPolicy = field(init=False)
    filter_policy: FilterPolicy = field(init=False)
    scorers: ScorerCollection = field(init=False)

    def __post_init__(self) -> None:
        self.stock = Stock()
        self.expansion_policy = ExpansionPolicy(self)
        self.filter_policy = FilterPolicy(self)
        self.scorers = ScorerCollection(self)
        self._logger = logger()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Configuration):
            return False
        for key, setting in vars(self).items():
            if isinstance(setting, (int, float, str, bool, list)):
                if (
                    vars(self)[key] != vars(other)[key]
                    or self.search != other.search
                    or self.post_processing != other.post_processing
                ):
                    return False
        return True

    @classmethod
    def from_dict(cls, source: StrDict) -> "Configuration":
        """
        从字典结构加载配置。

        字典中未显式设置的参数会沿用默认值。
        同时会直接加载配置中声明的策略和库存。

        :param source: 配置字典来源
        :return: 根据输入生成的 `Configuration` 对象
        """
        expansion_config = source.pop("expansion", {})
        filter_config = source.pop("filter", {})
        stock_config = source.pop("stock", {})
        scorer_config = source.pop("scorer", {})

        config_obj = Configuration()
        config_obj._update_from_config(dict(source))

        config_obj.expansion_policy.load_from_config(**expansion_config)
        config_obj.filter_policy.load_from_config(**filter_config)
        config_obj.stock.load_from_config(**stock_config)
        config_obj.scorers.create_default_scorers()
        config_obj.scorers.load_from_config(**scorer_config)

        return config_obj

    @classmethod
    def from_file(cls, filename: str) -> "Configuration":
        """
        从 yaml 文件加载配置。

        yaml 文件中未显式设置的参数会沿用默认值。
        文件中声明的策略和库存会被直接加载。
        参数值中也允许使用环境变量占位符。

        :param filename: yaml 文件路径
        :return: 从 yaml 文件构建得到的 `Configuration` 对象
        :raises:
            ValueError: 当参数值引用了当前环境中不存在的环境变量时抛出
        """
        with open(filename, "r") as fileobj:
            txt = fileobj.read()
        environ_var = re.findall(r"\$\{.+?\}", txt)
        for item in environ_var:
            if item[2:-1] not in os.environ:
                raise ValueError(f"'{item[2:-1]}' not in environment variables")
            txt = txt.replace(item, os.environ[item[2:-1]])
        _config = yaml.load(txt, Loader=yaml.SafeLoader)
        return Configuration.from_dict(_config)

    def _update_from_config(self, config: StrDict) -> None:
        self.post_processing = _PostprocessingConfiguration(
            **config.pop("post_processing", {})
        )

        search_config = config.pop("search", {})
        for setting, value in search_config.items():
            if value is None:
                continue
            if not hasattr(self.search, setting):
                raise AttributeError(f"Could not find attribute to set: {setting}")
            if setting.endswith("_bonds"):
                if not isinstance(value, list):
                    raise ValueError("Bond settings need to be lists")
                value = _handle_bond_pair_tuples(value) if value else []
            if setting == "algorithm_config":
                if not isinstance(value, dict):
                    raise ValueError("algorithm_config settings need to be dictionary")
                self.search.algorithm_config.update(value)
            else:
                setattr(self.search, setting, value)


def _handle_bond_pair_tuples(bonds: List[List[int]]) -> List[List[int]]:
    """规范化断裂键或冻结键配置中的键对列表。"""

    if not all(len(bond_pair) == 2 for bond_pair in bonds):
        raise ValueError("Lists of bond pairs to break/freeze should be of length 2")
    return [bond_pair[:2] for bond_pair in bonds]
