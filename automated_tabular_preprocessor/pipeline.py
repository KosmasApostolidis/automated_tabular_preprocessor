"""Pipeline orchestrator. Composes Steps; routes fit_transform / transform calls."""
from __future__ import annotations

import pandas as pd

from .steps.base import Step


class Pipeline:
    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        for step in self.steps:
            result = step.fit_transform(result)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        for step in self.steps:
            result = step.transform(result)
        return result
