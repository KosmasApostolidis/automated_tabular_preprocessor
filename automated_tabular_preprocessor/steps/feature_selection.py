"""Select top-K features. Strategy pattern: ExtraTrees importance or f_classif."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from sklearn.ensemble          import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from ..constants      import EXTRA_TREES_N_ESTIMATORS
from ..logging_config import get_logger
from .base            import MutableTargetSpec

logger = get_logger(__name__)


class FeatureSelectorStrategy(Protocol):
    def select(self, x: pd.DataFrame, y: pd.Series, k: int) -> list[str]: ...


class TreesSelector:
    def __init__(self, random_state: int, n_estimators: int = EXTRA_TREES_N_ESTIMATORS) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators

    def select(self, x: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
        model       = ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        model.fit(x, y)
        importances = pd.Series(model.feature_importances_, index=x.columns)
        return importances.nlargest(k).index.tolist()


class FClassifSelector:
    def select(self, x: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(x, y)
        return x.columns[selector.get_support()].tolist()


class FeatureSelectionStep:
    def __init__(
        self,
        strategy:    FeatureSelectorStrategy,
        target_spec: MutableTargetSpec,
        k:           int,
    ) -> None:
        self.strategy        = strategy
        self.target_spec     = target_spec
        self.k               = k
        self.columns_to_keep: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.k <= 0:
            return df
        target_columns_present = [c for c in self.target_spec.columns if c in df.columns]
        x = df.drop(columns=target_columns_present)
        y = self._target_series_for_selector(df, target_columns_present)
        self.columns_to_keep = self.strategy.select(x, y, self.k) + target_columns_present
        return self._project(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_to_keep:
            return df
        return self._project(df)

    def _project(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.columns_to_keep if c in df.columns]
        result  = df[present]
        logger.info("Feature selection kept %d columns.", result.shape[1])
        return result

    def _target_series_for_selector(self, df: pd.DataFrame, target_columns: list[str]) -> pd.Series:
        if len(target_columns) == 1:
            return df[target_columns[0]]
        return df[target_columns].values.argmax(axis=1)
