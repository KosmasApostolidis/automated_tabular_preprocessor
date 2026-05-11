"""Categorical feature encoding. Strategy pattern: one-hot or factorize."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..constants      import UNKNOWN_CATEGORY_CODE
from ..logging_config import get_logger

logger = get_logger(__name__)


class CategoricalEncoderStrategy(Protocol):
    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: ...
    def transform(self,     df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: ...


class OneHotEncoder:
    def __init__(self) -> None:
        self.columns_after_encoding: list[str] = []

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
        self.columns_after_encoding = result.columns.tolist()
        return result

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
        for col in self.columns_after_encoding:
            if col not in result.columns:
                result[col] = 0
        return result[self.columns_after_encoding]


class FactorizeEncoder:
    def __init__(self) -> None:
        self.encoding_mappings: dict[str, dict] = {}
        self.columns_after_encoding: list[str]  = []

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()
        for col in columns:
            codes, uniques            = pd.factorize(result[col])
            self.encoding_mappings[col] = {val: idx for idx, val in enumerate(uniques)}
            result[col]               = codes
        self.columns_after_encoding = result.columns.tolist()
        return result

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()
        for col in columns:
            mapping     = self.encoding_mappings.get(col, {})
            result[col] = result[col].map(mapping).fillna(UNKNOWN_CATEGORY_CODE).astype(int)
        return result


class CategoricalEncoderStep:
    def __init__(self, strategy: CategoricalEncoderStrategy, target_column: str | None) -> None:
        self.strategy                  = strategy
        self.target_column             = target_column
        self.encoded_categorical_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self._categorical_columns(df)
        if not columns:
            return df
        before                       = set(df.columns)
        result                       = self.strategy.fit_transform(df, columns)
        self.encoded_categorical_cols = (
            list(columns) if isinstance(self.strategy, FactorizeEncoder)
            else [c for c in result.columns if c not in before]
        )
        logger.info("Encoded categorical columns: %s", columns)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self._categorical_columns(df)
        if not columns and isinstance(self.strategy, FactorizeEncoder):
            return df
        if isinstance(self.strategy, OneHotEncoder):
            # one-hot must always run on test set to add missing learned columns.
            return self.strategy.transform(df, columns)
        return self.strategy.transform(df, columns)

    def _categorical_columns(self, df: pd.DataFrame) -> list[str]:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_column in columns:
            columns.remove(self.target_column)
        return columns

    @property
    def columns_after_encoding(self) -> list[str]:
        return self.strategy.columns_after_encoding

    @property
    def cat_encoding_mappings(self) -> dict:
        return getattr(self.strategy, "encoding_mappings", {})
