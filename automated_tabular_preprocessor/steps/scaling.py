"""Scale numeric features. Strategy pattern over sklearn scalers."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..logging_config import get_logger
from .base            import MutableTargetSpec, select_numeric_columns_excluding

logger = get_logger(__name__)


class ScalerAdapter(Protocol):
    scaler_object: object

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: ...
    def transform(self,     df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: ...


class _SklearnScalerAdapter:
    def __init__(self, scaler) -> None:
        self.scaler_object = scaler
        self._fitted       = False

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result          = df.copy()
        result[columns] = self.scaler_object.fit_transform(result[columns])
        self._fitted    = True
        return result

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not self._fitted:
            return df
        result          = df.copy()
        result[columns] = self.scaler_object.transform(result[columns])
        return result


class StandardScalerAdapter(_SklearnScalerAdapter):
    def __init__(self) -> None:
        super().__init__(StandardScaler())


class MinMaxScalerAdapter(_SklearnScalerAdapter):
    def __init__(self) -> None:
        super().__init__(MinMaxScaler())


class RobustScalerAdapter(_SklearnScalerAdapter):
    def __init__(self) -> None:
        super().__init__(RobustScaler())


class ScalingStep:
    def __init__(
        self,
        scaler:                  ScalerAdapter,
        target_spec:             MutableTargetSpec,
        features_to_skip_scaling: list[str],
    ) -> None:
        self.scaler                  = scaler
        self.target_spec             = target_spec
        self.features_to_skip_scaling = list(features_to_skip_scaling)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self._columns_to_scale(df)
        if not columns:
            return df
        return self.scaler.fit_transform(df, columns)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self._columns_to_scale(df)
        if not columns:
            return df
        return self.scaler.transform(df, columns)

    def _columns_to_scale(self, df: pd.DataFrame) -> list[str]:
        excluded = {self.target_spec.name, *self.target_spec.columns, *self.features_to_skip_scaling}
        return select_numeric_columns_excluding(df, excluded)

    @property
    def scaler_object(self):
        return self.scaler.scaler_object
