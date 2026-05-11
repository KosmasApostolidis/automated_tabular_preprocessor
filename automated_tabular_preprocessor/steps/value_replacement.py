"""Replace a sentinel value with the per-column mode learned at fit time."""
from __future__ import annotations

import pandas as pd

from ..logging_config import get_logger

logger = get_logger(__name__)


class ReplaceValueWithModeStep:
    def __init__(self, value_to_replace) -> None:
        self.value_to_replace       = value_to_replace
        self.mode_replacements: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._learn_modes(df)
        return self._apply_modes(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply_modes(df)

    def _learn_modes(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            if not (df[col] == self.value_to_replace).any():
                continue
            valid = df.loc[df[col] != self.value_to_replace, col]
            if not valid.empty:
                self.mode_replacements[col] = valid.mode()[0]

    def _apply_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, mode_val in self.mode_replacements.items():
            if col not in result.columns:
                continue
            mask  = result[col] == self.value_to_replace
            count = int(mask.sum())
            if count > 0:
                result.loc[mask, col] = mode_val
                logger.info("Replaced %d in '%s' with %r", count, col, mode_val)
        return result
