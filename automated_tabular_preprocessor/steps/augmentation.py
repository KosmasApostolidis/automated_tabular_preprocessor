"""Synthetic data augmentation. Strategy pattern: SMOTE, CTGAN, or TVAE."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..constants      import CTGAN_EPOCHS, TVAE_EPOCHS
from ..logging_config import get_logger

logger = get_logger(__name__)


class AugmenterStrategy(Protocol):
    def augment(self, df: pd.DataFrame, target_column: str, random_state: int) -> pd.DataFrame: ...


class SmoteAugmenter:
    def augment(self, df: pd.DataFrame, target_column: str, random_state: int) -> pd.DataFrame:
        from imblearn.over_sampling import SMOTE

        if df.select_dtypes(include=["object", "category"]).shape[1] > 0:
            logger.error("SMOTE requires all-numeric features; skipping.")
            return df
        x                  = df.drop(columns=[target_column])
        y                  = df[target_column]
        smote              = SMOTE(random_state=random_state)
        x_resampled, y_res = smote.fit_resample(x, y)
        return pd.concat(
            [pd.DataFrame(x_resampled, columns=x.columns), pd.Series(y_res, name=target_column)],
            axis=1,
        )


class _SynthesizerAugmenter:
    def __init__(self, factory, epochs: int) -> None:
        self._factory = factory
        self._epochs  = epochs

    def augment(self, df: pd.DataFrame, target_column: str, random_state: int) -> pd.DataFrame:
        synth = self._factory(epochs=self._epochs)
        synth.fit(df, discrete_columns=[target_column])
        synthetic = synth.sample(len(df))
        return pd.concat([df, synthetic], ignore_index=True)


class CtganAugmenter(_SynthesizerAugmenter):
    def __init__(self) -> None:
        from ctgan import CTGAN
        super().__init__(CTGAN, CTGAN_EPOCHS)


class TvaeAugmenter(_SynthesizerAugmenter):
    def __init__(self) -> None:
        from ctgan import TVAE
        super().__init__(TVAE, TVAE_EPOCHS)


class AugmentationStep:
    def __init__(
        self,
        strategy:      AugmenterStrategy,
        target_column: str,
        random_state:  int,
    ) -> None:
        self.strategy      = strategy
        self.target_column = target_column
        self.random_state  = random_state

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_column is None or self.target_column not in df.columns:
            logger.warning("No target column for augmentation; skipping.")
            return df
        logger.info("Augmenting via %s. Before: %s", type(self.strategy).__name__, df[self.target_column].value_counts().to_dict())
        result = self.strategy.augment(df, self.target_column, self.random_state)
        logger.info("After augmentation: %s", result[self.target_column].value_counts().to_dict())
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # augmentation is train-only.
