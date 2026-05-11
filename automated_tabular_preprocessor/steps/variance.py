"""Drop numeric columns whose variance falls below a threshold."""
from __future__ import annotations

import numpy  as np
import pandas as pd

from ..constants      import DEFAULT_VARIANCE_THRESHOLD
from ..logging_config import get_logger
from .base            import MutableTargetSpec

logger = get_logger(__name__)


class LowVarianceDropStep:
    def __init__(
        self,
        target_spec: MutableTargetSpec,
        threshold:   float = DEFAULT_VARIANCE_THRESHOLD,
    ) -> None:
        self.target_spec       = target_spec
        self.threshold         = threshold
        self.low_variance_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        protected             = {self.target_spec.name, *self.target_spec.columns}
        numeric               = df.select_dtypes(include=np.number).columns
        candidates            = [c for c in numeric if c not in protected]
        self.low_variance_cols = [c for c in candidates if df[c].var() < self.threshold]
        return self._drop(df, self.low_variance_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.low_variance_cols if c in df.columns]
        return self._drop(df, present)

    def _drop(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            return df
        logger.info("Dropping low-variance columns: %s", columns)
        return df.drop(columns=columns)
