"""Step protocol and small shared helpers."""
from __future__ import annotations

from dataclasses    import dataclass, field
from typing         import Iterable, Protocol, runtime_checkable

import numpy  as np
import pandas as pd


@runtime_checkable
class Step(Protocol):
    """A pipeline step learns from training data, then re-applies to new data."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def transform(self,     df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class TargetSpec:
    """Immutable description of the target column(s) after encoding."""

    name:    str
    columns: tuple[str, ...]  # final target columns (1 for label, N for one-hot)


@dataclass
class MutableTargetSpec:
    """Mutable target column reference shared between steps that need to skip target columns."""

    name:    str
    columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.columns:
            self.columns = [self.name]


def select_numeric_columns_excluding(
    df:       pd.DataFrame,
    excluded: Iterable[str],
) -> list[str]:
    excluded_set = set(excluded)
    return [c for c in df.select_dtypes(include=np.number).columns if c not in excluded_set]
