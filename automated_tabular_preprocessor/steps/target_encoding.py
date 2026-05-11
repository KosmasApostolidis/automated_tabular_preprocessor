"""Target column encoding. Strategy: label (classical) or one-hot (neural network)."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..constants      import UNKNOWN_CATEGORY_CODE
from ..logging_config import get_logger
from .base            import MutableTargetSpec

logger = get_logger(__name__)


class TargetEncoderStrategy(Protocol):
    target_was_encoded:      bool
    target_encoding_mapping: dict | None
    final_target_cols:       list[str]

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame: ...
    def transform(self,     df: pd.DataFrame, target_column: str) -> pd.DataFrame: ...


class LabelTargetEncoder:
    def __init__(self) -> None:
        self.target_was_encoded      = False
        self.target_encoding_mapping: dict | None = None
        self.final_target_cols:      list[str]   = []

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        if df[target_column].dtype not in ("object", "category"):
            logger.info("Target '%s' already numeric; skipping label encoding.", target_column)
            self.final_target_cols = [target_column]
            return df

        result                       = df.copy()
        codes, uniques               = pd.factorize(result[target_column])
        result[target_column]        = codes
        self.target_encoding_mapping = {val: idx for idx, val in enumerate(uniques)}
        self.target_was_encoded      = True
        self.final_target_cols       = [target_column]
        logger.info("Label-encoded target '%s' -> %s", target_column, self.target_encoding_mapping)
        return result

    def transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        if not self.target_was_encoded or target_column not in df.columns:
            return df
        result                = df.copy()
        result[target_column] = (
            result[target_column]
            .map(self.target_encoding_mapping)
            .fillna(UNKNOWN_CATEGORY_CODE)
            .astype(int)
        )
        return result


class OneHotTargetEncoder:
    def __init__(self) -> None:
        self.target_was_encoded      = False
        self.target_encoding_mapping: dict | None = None
        self.final_target_cols:      list[str]   = []

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        if df[target_column].dtype not in ("object", "category"):
            logger.info("Target '%s' already numeric; skipping one-hot target encoding.", target_column)
            self.final_target_cols = [target_column]
            return df
        original_cols          = set(df.columns)
        result                 = pd.get_dummies(df, columns=[target_column], prefix=target_column, dtype=int)
        self.final_target_cols = sorted(set(result.columns) - original_cols)
        self.target_was_encoded = True
        logger.info("One-hot encoded target '%s' -> %s", target_column, self.final_target_cols)
        return result

    def transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        if not self.target_was_encoded:
            return df
        original_cols = set(df.columns)
        result        = pd.get_dummies(df, columns=[target_column], prefix=target_column, dtype=int)
        for col in self.final_target_cols:
            if col not in result.columns:
                result[col] = 0
        extras = [c for c in (set(result.columns) - original_cols) if c not in self.final_target_cols]
        if extras:
            result = result.drop(columns=extras)
        return result


class TargetEncoderStep:
    def __init__(
        self,
        strategy:      TargetEncoderStrategy,
        target_column: str,
        target_spec:   MutableTargetSpec,
    ) -> None:
        self.strategy      = strategy
        self.target_column = target_column
        self.target_spec   = target_spec

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.strategy.fit_transform(df, self.target_column)
        self.target_spec.columns = list(self.strategy.final_target_cols)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.strategy.transform(df, self.target_column)
        self.target_spec.columns = list(self.strategy.final_target_cols)
        return result

    @property
    def target_was_encoded(self) -> bool:
        return self.strategy.target_was_encoded

    @property
    def target_encoding_mapping(self) -> dict | None:
        return self.strategy.target_encoding_mapping

    @property
    def final_target_cols(self) -> list[str]:
        return self.strategy.final_target_cols
