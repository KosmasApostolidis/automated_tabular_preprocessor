"""Drop columns highly correlated with another feature."""
from __future__ import annotations

import numpy  as np
import pandas as pd

from ..constants      import DEFAULT_CORRELATION_THRESHOLD
from ..logging_config import get_logger
from .base            import MutableTargetSpec

logger = get_logger(__name__)


class HighCorrelationDropStep:
    def __init__(
        self,
        target_spec: MutableTargetSpec,
        threshold:   float = DEFAULT_CORRELATION_THRESHOLD,
    ) -> None:
        self.target_spec            = target_spec
        self.threshold              = threshold
        self.highly_correlated_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        candidate_df                  = self._candidate_frame(df)
        self.highly_correlated_cols   = self._find_correlated(candidate_df)
        return self._drop(df, self.highly_correlated_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.highly_correlated_cols if c in df.columns]
        return self._drop(df, present)

    def _candidate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        protected = {self.target_spec.name, *self.target_spec.columns}
        numeric   = df.select_dtypes(include=np.number)
        return numeric.drop(columns=[c for c in protected if c in numeric.columns])

    def _find_correlated(self, df: pd.DataFrame) -> list[str]:
        if df.shape[1] == 0:
            return []
        corr_matrix = df.corr().abs()
        upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        return [col for col in upper.columns if any(upper[col] > self.threshold)]

    def _drop(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            return df
        logger.info("Dropping highly correlated columns: %s", columns)
        return df.drop(columns=columns)
