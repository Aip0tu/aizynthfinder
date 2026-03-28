"""实现多种扩展策略的模块。"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aizynthfinder.chem import SmilesBasedRetroReaction, TemplatedRetroReaction
from aizynthfinder.context.policy.utils import _make_fingerprint
from aizynthfinder.utils.exceptions import PolicyException
from aizynthfinder.utils.logging import logger
from aizynthfinder.utils.models import load_model

if TYPE_CHECKING:
    from aizynthfinder.chem import TreeMolecule
    from aizynthfinder.chem.reaction import RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import (
        Any,
        Dict,
        List,
        Optional,
        Sequence,
        StrDict,
        Tuple,
    )


class ExpansionStrategy(abc.ABC):
    """
    所有扩展策略的抽象基类。

    可以直接调用 `get_actions` 方法，
    也可以把实例当作可调用对象，对分子列表执行扩展。

    .. code-block::

        expander = MyExpansionStrategy("dummy", config)
        actions, priors = expander.get_actions(molecules)
        actions, priors = expander(molecules)

    :param key: 策略键或标签
    :param config: 树搜索配置
    """

    _required_kwargs: List[str] = []

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        if any(name not in kwargs for name in self._required_kwargs):
            raise PolicyException(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}"
            )
        self._config = config
        self._logger = logger()
        self.key = key

    def __call__(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        return self.get_actions(molecules, cache_molecules)

    @abc.abstractmethod
    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        获取一组分子的所有高概率动作。

        :param molecules: 待考虑的分子集合
        :param cache_molecules: 额外提交给扩展策略的分子，
            这些分子只会写入缓存，供后续使用
        :return: 动作列表及其对应先验概率
        """

    def reset_cache(self) -> None:
        """重置预测缓存。"""


class MultiExpansionStrategy(ExpansionStrategy):
    """
    组合多个扩展策略的基类。

    可以通过 `get_actions` 调用，也可以把实例直接当作可调用对象使用。

    :ivar expansion_strategy_keys: 选中的扩展策略键列表
    :ivar additive_expansion: 是否合并所有选中策略的动作与先验概率，
        默认为 `False`
    :ivar expansion_strategy_weights: 各扩展策略的权重列表，
        权重之和应为 1；若未设置，则默认每个策略权重都为 1

    :param key: 策略键或标签
    :param config: 树搜索配置
    :param expansion_strategies: 选中的扩展策略键列表，
        所有键都必须已在 `config` 的扩展策略配置中声明
    """

    _required_kwargs = ["expansion_strategies"]

    def __init__(
        self,
        key: str,
        config: Configuration,
        **kwargs: Any,
    ) -> None:
        super().__init__(key, config, **kwargs)
        self._config = config
        self._expansion_strategies: List[ExpansionStrategy] = []
        self.expansion_strategy_keys = kwargs["expansion_strategies"]

        self.cutoff_number = kwargs.get("cutoff_number")
        if self.cutoff_number:
            print(f"Setting multi-expansion cutoff_number: {self.cutoff_number}")

        self.expansion_strategy_weights = self._set_expansion_strategy_weights(kwargs)
        self.additive_expansion: bool = bool(kwargs.get("additive_expansion", False))
        self._logger.info(
            f"Multi-expansion strategy with policies: {self.expansion_strategy_keys}"
            f", and corresponding weights: {self.expansion_strategy_weights}"
        )

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        使用选定的策略，获取一组分子的所有高概率动作。

        默认实现会在 `additive_expansion=True` 时，
        将所有已选扩展策略的动作与先验概率分别合并为两个列表。
        子类可以重写该方法，以实现其他组合方式。

        :param molecules: 待考虑的分子集合
        :param cache_molecules: 额外提交给扩展策略的分子，
            这些分子只会写入缓存，供后续使用
        :return: 动作列表及其对应先验概率
        :raises: PolicyException: 当策略未被正确选中时抛出
        """
        expansion_strategies = self._get_expansion_strategies_from_config()

        all_possible_actions = []
        all_priors = []
        for expansion_strategy, expansion_strategy_weight in zip(
            expansion_strategies, self.expansion_strategy_weights
        ):
            possible_actions, priors = expansion_strategy.get_actions(
                molecules, cache_molecules
            )

            all_possible_actions.extend(possible_actions)
            if not self.additive_expansion and all_possible_actions:
                all_priors.extend(priors)
                break

            weighted_prior = [expansion_strategy_weight * p for p in priors]

            all_priors.extend(weighted_prior)

        all_possible_actions, all_priors = self._prune_actions(
            all_possible_actions, all_priors
        )
        return all_possible_actions, all_priors

    def _get_expansion_strategies_from_config(self) -> List[ExpansionStrategy]:
        if self._expansion_strategies:
            return self._expansion_strategies

        if not all(
            key in self._config.expansion_policy.items
            for key in self.expansion_strategy_keys
        ):
            raise ValueError(
                "The input expansion strategy keys must exist in the "
                "expansion policies listed in config"
            )
        self._expansion_strategies = [
            self._config.expansion_policy[key] for key in self.expansion_strategy_keys
        ]

        for expansion_strategy, weight in zip(
            self._expansion_strategies, self.expansion_strategy_weights
        ):
            if not getattr(expansion_strategy, "rescale_prior", True) and weight < 1:
                setattr(expansion_strategy, "rescale_prior", True)
                self._logger.info(
                    f"Enforcing {expansion_strategy.key}.rescale_prior=True"
                )
        return self._expansion_strategies

    def _prune_actions(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        当指定最大动作数时，对动作进行裁剪。

        :param actions: 预测动作列表
        :param priors: 预测概率列表
        :return: 概率最高的 `self.cutoff_number` 个动作及其先验概率
        """
        if not self.cutoff_number:
            return actions, priors

        sortidx = np.argsort(np.array(priors))[::-1].astype(int)
        priors = [priors[idx] for idx in sortidx[0 : self.cutoff_number]]
        actions = [actions[idx] for idx in sortidx[0 : self.cutoff_number]]
        return actions, priors

    def _set_expansion_strategy_weights(self, kwargs: StrDict) -> List[float]:
        """
        根据配置中的 `kwargs` 设置各扩展策略权重。

        配置中的权重之和应为 1。
        如果配置文件未设置，则默认每个策略的权重都为 1，
        以兼容旧版本行为。

        :param kwargs: `MultiExpansionStrategy` 的输入参数
        :raises: ValueError: 当配置中的权重和不为 1 时抛出
        :return: 扩展策略权重列表
        """
        if not "expansion_strategy_weights" in kwargs:
            return [1.0 for _ in self.expansion_strategy_keys]

        expansion_strategy_weights = kwargs["expansion_strategy_weights"]
        sum_weights = sum(expansion_strategy_weights)

        if sum_weights != 1:
            raise ValueError(
                "The expansion strategy weights in MultiExpansion should "
                "sum to one. -> "
                f"sum({expansion_strategy_weights})={sum_weights}."
            )

        return expansion_strategy_weights


class TemplateBasedExpansionStrategy(ExpansionStrategy):
    """
    基于模板的扩展策略，展开后返回 `TemplatedRetroReaction` 对象。

    :ivar template_column: 模板文件中存放模板内容的列名
    :ivar cutoff_cumulative: 建议模板的累计概率阈值
    :ivar cutoff_number: 返回模板的最大数量
    :ivar use_rdchiral: 是否使用 RDChiral 应用模板
    :ivar use_remote_models: 是否连接远程 TensorFlow 服务
    :ivar rescale_prior: 是否对先验概率再次应用 softmax
    :ivar chiral_fingerprints: 若为 `True`，则使用手性指纹进行扩展
    :ivar mask: 反应模板掩码向量，长度应与模板数一致；
        如果未提供掩码文件，则为 `None`

    :param key: 策略键或标签
    :param config: 树搜索配置
    :param model: 策略模型来源
    :param template: 包含模板的 HDF5 文件路径
    :raises PolicyException: 当模型输出向量长度与模板数量不一致时抛出
    """

    _required_kwargs = [
        "model",
        "template",
    ]

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)

        source = kwargs["model"]
        templatefile = kwargs["template"]
        maskfile: str = kwargs.get("mask", "")
        self.template_column: str = kwargs.get("template_column", "retro_template")
        self.cutoff_cumulative: float = float(kwargs.get("cutoff_cumulative", 0.995))
        self.cutoff_number: int = int(kwargs.get("cutoff_number", 50))
        self.use_rdchiral: bool = bool(kwargs.get("use_rdchiral", True))
        self.use_remote_models: bool = bool(kwargs.get("use_remote_models", False))
        self.rescale_prior: bool = bool(kwargs.get("rescale_prior", False))
        self.chiral_fingerprints = bool(kwargs.get("chiral_fingerprints", False))

        self._logger.info(
            f"Loading template-based expansion policy model from {source} to {self.key}"
        )
        self.model = load_model(source, self.key, self.use_remote_models)

        self._logger.info(f"Loading templates from {templatefile} to {self.key}")
        if templatefile.endswith(".csv.gz") or templatefile.endswith(".csv"):
            self.templates: pd.DataFrame = pd.read_csv(
                templatefile, index_col=0, sep="\t"
            )
        else:
            self.templates = pd.read_hdf(templatefile, "table")

        self.mask: Optional[np.ndarray] = (
            self._load_mask_file(maskfile) if maskfile else None
        )

        if hasattr(self.model, "output_size") and len(self.templates) != self.model.output_size:  # type: ignore
            raise PolicyException(
                f"The number of templates ({len(self.templates)}) does not agree with the "  # type: ignore
                f"output dimensions of the model ({self.model.output_size})"
            )
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        在给定截断条件下，获取一组分子的所有高概率动作。

        :param molecules: 待考虑的分子集合
        :param cache_molecules: 额外提交给扩展策略的分子，
            这些分子只会写入缓存，供后续使用
        :return: 动作列表及其对应先验概率
        """

        possible_actions = []
        priors: List[float] = []
        cache_molecules = cache_molecules or []
        self._update_cache(list(molecules) + list(cache_molecules))

        for mol in molecules:
            probable_transforms_idx, probs = self._cache[mol.inchi_key]
            possible_moves = self.templates.iloc[probable_transforms_idx]
            if self.rescale_prior:
                probs /= probs.sum()
            priors.extend(probs)
            for idx, (move_index, move) in enumerate(possible_moves.iterrows()):
                metadata = dict(move)
                del metadata[self.template_column]
                metadata["policy_probability"] = float(probs[idx].round(4))
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key
                metadata["template_code"] = move_index
                metadata["template"] = move[self.template_column]
                possible_actions.append(
                    TemplatedRetroReaction(
                        mol,
                        smarts=move[self.template_column],
                        metadata=metadata,
                        use_rdchiral=self.use_rdchiral,
                    )
                )
        return possible_actions, priors  # type: ignore

    def reset_cache(self) -> None:
        """重置预测缓存。"""
        self._cache = {}

    def _cutoff_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        根据截断条件选出得分最高的模板变换。

        选择规则为：
            * 累计概率不超过阈值 `cutoff_cumulative`
            * 或最多保留 `cutoff_number` 个结果
        """
        if self.mask is not None:
            predictions[~self.mask] = 0
        sortidx = np.argsort(predictions)[::-1]
        cumsum: np.ndarray = np.cumsum(predictions[sortidx])
        if any(cumsum >= self.cutoff_cumulative):
            maxidx = int(np.argmin(cumsum < self.cutoff_cumulative))
        else:
            maxidx = len(cumsum)
        maxidx = min(maxidx, self.cutoff_number) or 1
        return sortidx[:maxidx]

    def _load_mask_file(self, maskfile: str) -> np.ndarray:
        self._logger.info(f"Loading masking of templates from {maskfile} to {self.key}")
        mask = np.load(maskfile)["arr_0"]
        if len(mask) != len(self.templates):
            raise PolicyException(
                f"The number of masks {len(mask)} does not match the number of templates {len(self.templates)}"
            )
        return mask

    def _update_cache(self, molecules: Sequence[TreeMolecule]) -> None:
        pred_inchis = []
        fp_list = []
        for molecule in molecules:
            if molecule.inchi_key in self._cache or molecule.inchi_key in pred_inchis:
                continue
            fp_list.append(
                _make_fingerprint(molecule, self.model, self.chiral_fingerprints)
            )
            pred_inchis.append(molecule.inchi_key)

        if not pred_inchis:
            return

        pred_list = np.asarray(self.model.predict(np.vstack(fp_list)))
        for pred, inchi in zip(pred_list, pred_inchis):
            probable_transforms_idx = self._cutoff_predictions(pred)
            self._cache[inchi] = (
                probable_transforms_idx,
                pred[probable_transforms_idx],
            )


class TemplateBasedDirectExpansionStrategy(TemplateBasedExpansionStrategy):
    """
    直接应用模板的扩展策略，展开后返回 `SmilesBasedRetroReaction` 对象。

    :param key: 策略键或标签
    :param config: 树搜索配置
    :param source: 策略模型来源
    :param templatefile: 包含模板的 HDF5 文件路径
    :raises PolicyException: 当模型输出向量长度与模板数量不一致时抛出
    """

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        在给定截断条件下，获取一组分子的所有高概率动作。

        :param molecules: 待考虑的分子集合
        :param cache_molecules: 额外提交给扩展策略的分子，
            这些分子只会写入缓存，供后续使用
        :return: 动作列表及其对应先验概率
        """
        possible_actions = []
        priors = []

        super_actions, super_priors = super().get_actions(molecules, cache_molecules)
        for templated_action, prior in zip(super_actions, super_priors):
            for reactants in templated_action.reactants:
                reactants_str = ".".join(mol.smiles for mol in reactants)
                new_action = SmilesBasedRetroReaction(
                    templated_action.mol,
                    metadata=templated_action.metadata,
                    reactants_str=reactants_str,
                )
                possible_actions.append(new_action)
                priors.append(prior)

        return possible_actions, priors  # type: ignore
