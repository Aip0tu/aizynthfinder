"""定义各类集合对象基类的模块，例如库存、策略和评分器集合。"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.utils.type_utils import Any, List, Optional, StrDict, Union


class ContextCollection(abc.ABC):
    """
    可加载并支持选择/取消选择操作的集合抽象基类。

    可以通过以下方式获取单个条目：

    .. code-block::

        an_item = collection["key"]

    也可以通过以下方式删除条目：

    .. code-block::

        del collection["key"]


    """

    _single_selection = False
    _collection_name = "collection"

    def __init__(self) -> None:
        self._items: StrDict = {}
        self._selection: List[str] = []
        self._logger = logger()

    def __delitem__(self, key: str) -> None:
        if key not in self._items:
            raise KeyError(
                f"{self._collection_name.capitalize()} with name {key} not loaded."
            )
        del self._items[key]

    def __getitem__(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(
                f"{self._collection_name.capitalize()} with name {key} not loaded."
            )
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[str]:
        """返回当前可用条目的键列表。"""
        return list(self._items.keys())

    @property
    def selection(self) -> Union[List[str], str, None]:
        """返回当前被选中的条目键。"""
        if self._single_selection:
            return self._selection[0] if self._selection else None
        return self._selection

    @selection.setter
    def selection(self, value: str) -> None:
        self.select(value)

    def deselect(self, key: Optional[str] = None) -> None:
        """
        取消一个或全部条目的选择状态。

        如果未传入键，则会取消全部选择。

        :param key: 要取消选择的条目键，默认为 `None`
        :raises KeyError: 当指定键当前并未被选中时抛出
        """
        if not key:
            self._selection = []
            return

        if key not in self._selection:
            raise KeyError(f"Cannot deselect {key} because it is not selected")
        self._selection.remove(key)

    @abc.abstractmethod
    def load(self, *_: Any) -> None:
        """加载单个条目，由子类实现。"""

    @abc.abstractmethod
    def load_from_config(self, **config: Any) -> None:
        """根据配置加载条目，由子类实现。"""

    def select(self, value: Union[str, List[str]], append: bool = False) -> None:
        """
        选择一个或多个条目。

        对于单选集合，只接受单个值。
        对于多选集合，默认会覆盖当前选择；
        只有在 `append=True` 且传入单个键时才会追加。

        :param value: 要选择的条目键或键列表
        :param append: 若为 `True`，则把单个键追加到当前选择中
        :raises ValueError: 当单选集合却传入多个键时抛出
        :raises KeyError: 当任意一个键未对应已加载条目时抛出
        """
        if self._single_selection and not isinstance(value, str) and len(value) > 1:
            raise ValueError(f"Cannot select more than one {self._collection_name}")

        keys = [value] if isinstance(value, str) else value

        for key in keys:
            if key not in self._items:
                raise KeyError(
                    f"Invalid key specified {key} when selecting {self._collection_name}"
                )

        if self._single_selection:
            self._selection = [keys[0]]
        elif isinstance(value, str) and append:
            self._selection.append(value)
        else:
            self._selection = list(keys)

        self._logger.info(f"Selected as {self._collection_name}: {', '.join(keys)}")

    def select_all(self) -> None:
        """选择所有已加载条目。"""
        if self.items:
            self.select(self.items)

    def select_first(self) -> None:
        """选择第一个已加载条目。"""
        if self.items:
            self.select(self.items[0])

    def select_last(self) -> None:
        """选择最后一个已加载条目。"""
        if self.items:
            self.select(self.items[-1])
