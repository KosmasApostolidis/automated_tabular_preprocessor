"""Remove outlier rows via IQR. Train-time only; transform is identity."""
from __future__ import annotations

import pandas as pd

from ..constants      import DEFAULT_IQR_MULTIPLIER, IQR_LOWER_QUANTILE, IQR_UPPER_QUANTILE
from ..logging_config import get_logger
from .base            import MutableTargetSpec, select_numeric_columns_excluding

logger = get_logger(__name__)


class IqrOutlierRemovalStep:
    def __init__(
        self,
        target_spec:                MutableTargetSpec,
        encoded_categorical_columns: list[str],
        multiplier:                  float = DEFAULT_IQR_MULTIPLIER,
    ) -> None:
        self.target_spec                 = target_spec
        self.encoded_categorical_columns = encoded_categorical_columns
        self.multiplier                  = multiplier

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        excluded         = {self.target_spec.name, *self.encoded_categorical_columns}
        numeric_columns  = select_numeric_columns_excluding(df, excluded)
        if not numeric_columns:
            logger.info("No numeric columns for outlier removal; skipping.")
            return df

        outlier_indices: set = set()
        for col in numeric_columns:
            outlier_indices.update(self._outlier_indices_for_column(df, col))

        if not outlier_indices:
            return df
        result = df.drop(index=list(outlier_indices))
        logger.info("Removed %d outlier rows; new shape %s", len(outlier_indices), result.shape)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # row removal must not affect test set.

    def _outlier_indices_for_column(self, df: pd.DataFrame, col: str):
        lower_quartile = df[col].quantile(IQR_LOWER_QUANTILE)
        upper_quartile = df[col].quantile(IQR_UPPER_QUANTILE)
        iqr            = upper_quartile - lower_quartile
        lower_bound    = lower_quartile - self.multiplier * iqr
        upper_bound    = upper_quartile + self.multiplier * iqr
        return df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
