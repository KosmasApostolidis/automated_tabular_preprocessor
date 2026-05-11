"""Backward-compatible facade. Composes step objects from `.steps.*` and exposes the original API.

The public surface (class names, constructor kwargs, public attribute names, and the
`_xxx`-prefixed step methods consumed by the existing test suite) is preserved.
Internal mechanics now live in single-responsibility step classes.
"""
from __future__ import annotations

from abc            import ABC, abstractmethod
from collections    import Counter

import pandas as pd

from .constants                  import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_MISSING_VALUES_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_VARIANCE_THRESHOLD,
)
from .logging_config             import get_logger
from .pipeline                   import Pipeline
from .steps.augmentation         import (
    AugmentationStep,
    CtganAugmenter,
    SmoteAugmenter,
    TvaeAugmenter,
)
from .steps.base                 import MutableTargetSpec
from .steps.correlation          import HighCorrelationDropStep
from .steps.drop_columns         import DropColumnsStep, DropDuplicatesStep
from .steps.encoding             import (
    CategoricalEncoderStep,
    FactorizeEncoder,
    OneHotEncoder,
)
from .steps.feature_selection    import (
    FClassifSelector,
    FeatureSelectionStep,
    TreesSelector,
)
from .steps.missing_values       import MissingValueHandlerStep
from .steps.outliers             import IqrOutlierRemovalStep
from .steps.scaling              import (
    MinMaxScalerAdapter,
    RobustScalerAdapter,
    ScalingStep,
    StandardScalerAdapter,
)
from .steps.target_encoding      import (
    LabelTargetEncoder,
    OneHotTargetEncoder,
    TargetEncoderStep,
)
from .steps.value_replacement    import ReplaceValueWithModeStep
from .steps.variance             import LowVarianceDropStep

logger = get_logger(__name__)


_CATEGORICAL_ENCODER_REGISTRY = {
    "onehot":    OneHotEncoder,
    "factorize": FactorizeEncoder,
}

_SCALER_REGISTRY = {
    "standard": StandardScalerAdapter,
    "minmax":   MinMaxScalerAdapter,
    "robust":   RobustScalerAdapter,
}

_FEATURE_SELECTOR_REGISTRY = {
    "trees":     lambda rs: TreesSelector(random_state=rs),
    "f_classif": lambda rs: FClassifSelector(),
}

_AUGMENTER_REGISTRY = {
    "smote": SmoteAugmenter,
    "ctgan": CtganAugmenter,
    "tvae":  TvaeAugmenter,
}


class AbstractPreprocessor(ABC):
    def __init__(
        self,
        df                       = None,
        target_column            = None,
        cols_to_drop             = None,
        value_to_replace         = None,
        missing_values_threshold = DEFAULT_MISSING_VALUES_THRESHOLD,
        random_state             = DEFAULT_RANDOM_STATE,
        **kwargs,
    ) -> None:
        self.df                       = df
        self.target_column            = target_column
        self.cols_to_drop             = list(cols_to_drop) if cols_to_drop else []
        self.value_to_replace         = value_to_replace
        self.missing_values_threshold = missing_values_threshold
        self.random_state             = random_state

    def _check_class_balance(self, y, title: str = "") -> "AbstractPreprocessor":
        if isinstance(y, pd.DataFrame):
            target_series = y[self.target_column]
        elif isinstance(y, pd.Series):
            target_series = y
        else:
            raise TypeError(f"Expected a pandas DataFrame or Series, but got {type(y)}")

        for label, count in Counter(target_series).items():
            percentage = (count / len(target_series)) * 100
            logger.info("  Class %s: %d samples (%.2f%%)", label, count, percentage)
        return self

    @abstractmethod
    def run_preprocessing_pipeline(self, is_train: bool = True):
        raise NotImplementedError


class TabularDataPreprocessor(AbstractPreprocessor):
    def __init__(
        self,
        number_of_top_k_features = 0,
        features_to_skip_scaling = None,
        numerical_scaler         = "standard",
        categorical_encoder      = "onehot",
        model_type               = "classical",
        feature_selection_method = "trees",
        iqr_multiplier           = DEFAULT_IQR_MULTIPLIER,
        remove_highly_correlated = False,
        remove_low_variance      = False,
        remove_outliers          = False,
        encode_categorical       = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.number_of_top_k_features = number_of_top_k_features
        self.features_to_skip_scaling = list(features_to_skip_scaling) if features_to_skip_scaling else []
        self.numerical_scaler         = numerical_scaler
        self.categorical_encoder      = categorical_encoder
        self.model_type               = model_type
        self.feature_selection_method = feature_selection_method
        self.iqr_multiplier           = iqr_multiplier
        self.remove_highly_correlated = remove_highly_correlated
        self.remove_low_variance      = remove_low_variance
        self.remove_outliers          = remove_outliers
        self.encode_categorical       = encode_categorical

        self._target_spec             = MutableTargetSpec(name=self.target_column or "")
        self._drop_columns_step       = DropColumnsStep(self.cols_to_drop)
        self._drop_duplicates_step    = DropDuplicatesStep()
        self._missing_values_step     = MissingValueHandlerStep(threshold=self.missing_values_threshold)
        self._replace_value_step      = (
            ReplaceValueWithModeStep(self.value_to_replace) if self.value_to_replace is not None else None
        )
        self._outliers_step           = IqrOutlierRemovalStep(
            target_spec                 = self._target_spec,
            encoded_categorical_columns = [],
            multiplier                  = iqr_multiplier,
        )
        self._categorical_encoder_step = CategoricalEncoderStep(
            strategy      = self._build_categorical_encoder(),
            target_column = self.target_column,
        )
        self._target_encoder_step     = TargetEncoderStep(
            strategy      = self._build_target_encoder(),
            target_column = self.target_column,
            target_spec   = self._target_spec,
        )
        self._correlation_step        = HighCorrelationDropStep(self._target_spec)
        self._variance_step           = LowVarianceDropStep(self._target_spec)
        self._feature_selection_step  = FeatureSelectionStep(
            strategy    = self._build_feature_selector(),
            target_spec = self._target_spec,
            k           = self.number_of_top_k_features,
        )
        self._scaling_step:            ScalingStep | None = None

    # ----- step factories (single dispatch point for stringly-typed config) -----

    def _build_categorical_encoder(self):
        try:
            return _CATEGORICAL_ENCODER_REGISTRY[self.categorical_encoder]()
        except KeyError as exc:
            raise ValueError(f"Unknown categorical_encoder '{self.categorical_encoder}'.") from exc

    def _build_target_encoder(self):
        return OneHotTargetEncoder() if self.model_type == "neural_network" else LabelTargetEncoder()

    def _build_feature_selector(self):
        try:
            return _FEATURE_SELECTOR_REGISTRY[self.feature_selection_method](self.random_state)
        except KeyError as exc:
            raise ValueError(f"Unknown feature_selection_method '{self.feature_selection_method}'.") from exc

    def _build_scaler(self):
        try:
            return _SCALER_REGISTRY[self.numerical_scaler]()
        except KeyError as exc:
            raise ValueError(
                f"Unknown numerical_scaler '{self.numerical_scaler}'. "
                f"Valid options: 'standard', 'minmax', 'robust'."
            ) from exc

    # ----- backward-compatible private step API used directly by tests -----

    def _drop_columns(self):
        return self._apply(self._drop_columns_step.fit_transform)

    def _drop_duplicates(self):
        return self._apply(self._drop_duplicates_step.fit_transform)

    def _handle_missing_values(
        self,
        drop_high_missing: bool = True,
        numerical_imputer: str  = "median",
        is_train:          bool = True,
    ):
        self._missing_values_step.drop_high_missing = drop_high_missing
        self._missing_values_step.numeric_imputer   = numerical_imputer
        return self._apply_for_phase(self._missing_values_step, is_train)

    def _replace_value_with_mode(self, is_train: bool = True):
        if self._replace_value_step is None:
            return self
        return self._apply_for_phase(self._replace_value_step, is_train)

    def _remove_outliers_iqr(self):
        self._outliers_step.encoded_categorical_columns = self.encoded_categorical_cols
        return self._apply(self._outliers_step.fit_transform)

    def _encode_categorical_features(self, is_train: bool = True):
        return self._apply_for_phase(self._categorical_encoder_step, is_train)

    def _encode_target_column(self, is_train: bool = True):
        return self._apply_for_phase(self._target_encoder_step, is_train)

    def _remove_highly_correlated_features(
        self,
        threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        is_train:  bool  = True,
    ):
        self._correlation_step.threshold = threshold
        return self._apply_for_phase(self._correlation_step, is_train)

    def _remove_low_variance_features(
        self,
        threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        is_train:  bool  = True,
    ):
        self._variance_step.threshold = threshold
        return self._apply_for_phase(self._variance_step, is_train)

    def _select_top_k_features(self, is_train: bool = True):
        self._feature_selection_step.k = self.number_of_top_k_features
        return self._apply_for_phase(self._feature_selection_step, is_train)

    def _scale_numerical_features(self, is_train: bool = True):
        return self._apply_for_phase(self._get_scaling_step(), is_train)

    def _get_scaling_step(self) -> ScalingStep:
        if self._scaling_step is None:
            self._scaling_step = ScalingStep(
                scaler                   = self._build_scaler(),
                target_spec              = self._target_spec,
                features_to_skip_scaling = self.features_to_skip_scaling,
            )
        return self._scaling_step

    def _post_target_encoding_hook(self, is_train: bool = True):
        return self

    # ----- public entry point -----

    def run_preprocessing_pipeline(self, is_train: bool = True):
        pipeline = Pipeline(self._build_pipeline_steps())
        self.df  = pipeline.fit_transform(self.df) if is_train else pipeline.transform(self.df)
        return self

    def _build_pipeline_steps(self) -> list:
        steps = [self._drop_columns_step, self._drop_duplicates_step, self._missing_values_step]
        if self._replace_value_step is not None:
            steps.append(self._replace_value_step)
        if self.remove_outliers:
            steps.append(self._outlier_with_current_encoded_cols())
        if self.encode_categorical:
            steps.append(self._categorical_encoder_step)
        steps.append(self._target_encoder_step)
        steps.extend(self._post_target_steps())
        if self.remove_highly_correlated:
            steps.append(self._correlation_step)
        if self.remove_low_variance:
            steps.append(self._variance_step)
        if self.number_of_top_k_features > 0:
            self._feature_selection_step.k = self.number_of_top_k_features
            steps.append(self._feature_selection_step)
        steps.append(self._get_scaling_step())
        return steps

    def _outlier_with_current_encoded_cols(self) -> IqrOutlierRemovalStep:
        self._outliers_step.encoded_categorical_columns = self.encoded_categorical_cols
        return self._outliers_step

    def _post_target_steps(self) -> list:
        """Override hook for subclasses to inject steps after target encoding."""
        return []

    # ----- helpers -----

    def _apply(self, fn) -> "TabularDataPreprocessor":
        self.df = fn(self.df)
        return self

    def _apply_for_phase(self, step, is_train: bool) -> "TabularDataPreprocessor":
        fn      = step.fit_transform if is_train else step.transform
        self.df = fn(self.df)
        return self

    # ----- properties: bridge step state for backward-compatible attribute access -----

    @property
    def imputation_values(self) -> dict:
        return self._missing_values_step.imputation_values

    @property
    def high_missing_cols_dropped(self) -> list[str]:
        return self._missing_values_step.high_missing_cols_dropped

    @property
    def mode_replacements(self) -> dict:
        return self._replace_value_step.mode_replacements if self._replace_value_step else {}

    @property
    def scaler_object(self):
        return self._scaling_step.scaler_object if self._scaling_step else None

    @property
    def columns_to_keep(self):
        keep = self._feature_selection_step.columns_to_keep
        return keep if keep else None

    @property
    def cat_encoding_mappings(self) -> dict:
        return self._categorical_encoder_step.cat_encoding_mappings

    @property
    def encoded_categorical_cols(self) -> list[str]:
        return self._categorical_encoder_step.encoded_categorical_cols

    @property
    def columns_after_encoding(self) -> list[str]:
        return self._categorical_encoder_step.columns_after_encoding

    @property
    def target_encoding_mapping(self):
        return self._target_encoder_step.target_encoding_mapping

    @property
    def target_was_encoded(self) -> bool:
        return self._target_encoder_step.target_was_encoded

    @property
    def final_target_cols(self) -> list[str]:
        return self._target_encoder_step.final_target_cols or [self.target_column]

    @property
    def highly_correlated_cols(self) -> list[str]:
        return self._correlation_step.highly_correlated_cols

    @property
    def low_variance_cols(self) -> list[str]:
        return self._variance_step.low_variance_cols

    @property
    def top_k_features_columns(self):
        return self._feature_selection_step.columns_to_keep or None


class AugmentedDataPreprocessor(TabularDataPreprocessor):
    def __init__(self, *args, strategy: str = "smote", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.strategy           = strategy
        self._augmentation_step = AugmentationStep(
            strategy      = self._build_augmenter(),
            target_column = self.target_column,
            random_state  = self.random_state,
        )
        logger.info("Initialized AugmentedDataPreprocessor with strategy: %s", self.strategy.upper())

    def _build_augmenter(self):
        try:
            return _AUGMENTER_REGISTRY[self.strategy]()
        except KeyError:
            logger.error("Unknown augmentation strategy: %s", self.strategy)
            return _IdentityAugmenter()

    def _augment_data(self):
        return self._apply(self._augmentation_step.fit_transform)

    def _post_target_steps(self) -> list:
        return [self._augmentation_step]


class _IdentityAugmenter:
    def augment(self, df: pd.DataFrame, target_column: str, random_state: int) -> pd.DataFrame:
        return df
