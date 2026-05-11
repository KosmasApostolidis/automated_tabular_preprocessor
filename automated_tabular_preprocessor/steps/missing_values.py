"""Impute missing values; learn fill values from training data only."""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ..constants      import DEFAULT_MISSING_VALUES_THRESHOLD
from ..logging_config import get_logger

logger          = get_logger(__name__)
NumericImputer  = Literal["median", "mean"]


class MissingValueHandlerStep:
    def __init__(
        self,
        drop_high_missing: bool         = True,
        threshold:         float        = DEFAULT_MISSING_VALUES_THRESHOLD,
        numeric_imputer:   NumericImputer = "median",
    ) -> None:
        self.drop_high_missing          = drop_high_missing
        self.threshold                  = threshold
        self.numeric_imputer            = numeric_imputer
        self.imputation_values:    dict = {}
        self.high_missing_cols_dropped: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self._drop_high_missing_columns(df)
        self._learn_imputation_values(result)
        return self._apply_imputation(result)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.isnull().sum().sum() > 0:
            logger.warning("Missing values detected in test set; using training imputation values.")
        result = self._drop_remembered_columns(df)
        return self._apply_imputation(result)

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.drop_high_missing:
            return df
        missing_fraction = df.isnull().sum() / len(df)
        over_threshold   = missing_fraction[missing_fraction > self.threshold].index.tolist()
        self.high_missing_cols_dropped = over_threshold
        if not over_threshold:
            return df
        logger.info("Dropping columns over missing threshold: %s", over_threshold)
        return df.drop(columns=over_threshold)

    def _drop_remembered_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.high_missing_cols_dropped if c in df.columns]
        return df.drop(columns=present) if present else df

    def _learn_imputation_values(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if not df[col].isnull().any():
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                self.imputation_values[col] = self._numeric_fill_value(df[col])
            elif not df[col].mode().empty:
                self.imputation_values[col] = df[col].mode()[0]

    def _numeric_fill_value(self, series: pd.Series):
        return series.median() if self.numeric_imputer == "median" else series.mean()

    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, val in self.imputation_values.items():
            if col in result.columns and result[col].isnull().any():
                result[col] = result[col].fillna(val)
                logger.info("Filled '%s' with %r", col, val)
        return result
