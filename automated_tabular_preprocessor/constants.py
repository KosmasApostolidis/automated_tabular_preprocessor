"""Named constants. Replaces magic numbers scattered through the original module."""
from __future__ import annotations

DEFAULT_MISSING_VALUES_THRESHOLD: float = 0.4
DEFAULT_CORRELATION_THRESHOLD:    float = 0.8
DEFAULT_VARIANCE_THRESHOLD:       float = 0.01
DEFAULT_IQR_MULTIPLIER:           float = 1.5

IQR_LOWER_QUANTILE: float = 0.25
IQR_UPPER_QUANTILE: float = 0.75

EXTRA_TREES_N_ESTIMATORS: int = 50
CTGAN_EPOCHS:             int = 300
TVAE_EPOCHS:              int = 300

DEFAULT_RANDOM_STATE: int = 42

UNKNOWN_CATEGORY_CODE: int = -1
