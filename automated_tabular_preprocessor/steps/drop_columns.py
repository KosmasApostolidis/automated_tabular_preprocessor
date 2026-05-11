"""Drop user-specified columns and duplicate rows."""
from __future__ import annotations

import pandas as pd

from ..logging_config import get_logger

logger = get_logger(__name__)


class DropColumnsStep:
    def __init__(self, cols_to_drop: list[str]) -> None:
        self.cols_to_drop = list(cols_to_drop)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._drop(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._drop(df)

    def _drop(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cols_to_drop:
            logger.info("No columns to drop.")
            return df
        result = df.drop(columns=self.cols_to_drop, errors="ignore")
        logger.info("Dropped columns %s; new shape %s", self.cols_to_drop, result.shape)
        return result


class DropDuplicatesStep:
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._drop(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._drop(df)

    def _drop(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        result = df.drop_duplicates()
        logger.info("Dropped %d duplicate rows.", before - len(result))
        return result
