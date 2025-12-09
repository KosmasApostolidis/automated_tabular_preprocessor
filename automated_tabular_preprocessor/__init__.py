"""
Automated Tabular Preprocessing Library

A comprehensive library for preprocessing tabular data with support for
missing value handling, categorical encoding, feature selection, scaling,
and data augmentation.
"""

from .preprocessor import (
    AbstractPreprocessor,
    TabularDataPreprocessor,
    AugmentedDataPreprocessor,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "AbstractPreprocessor",
    "TabularDataPreprocessor",
    "AugmentedDataPreprocessor",
]
