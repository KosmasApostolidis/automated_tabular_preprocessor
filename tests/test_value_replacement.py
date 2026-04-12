import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestValueReplacement:
    def test_replaces_placeholder_with_mode(self):
        df = pd.DataFrame({
            'col': [1, 2, 2, 3, -999, -999, 2],
            'target': [0, 1, 0, 1, 0, 1, 0],
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
            value_to_replace=-999,
        )
        p._replace_value_with_mode(is_train=True)
        assert (p.df['col'] == -999).sum() == 0
        assert p.mode_replacements['col'] == 2

    def test_test_set_uses_train_mode(self):
        train = pd.DataFrame({
            'col': [1, 2, 2, -999],
            'target': [0, 1, 0, 1],
        })
        test = pd.DataFrame({
            'col': [5, -999, 7],
            'target': [0, 1, 0],
        })
        p = TabularDataPreprocessor(
            df=train.copy(),
            target_column='target',
            value_to_replace=-999,
        )
        p._replace_value_with_mode(is_train=True)

        p.df = test.copy()
        p._replace_value_with_mode(is_train=False)
        assert p.df['col'].iloc[1] == 2

    def test_no_replacement_when_none(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            value_to_replace=None,
        )
        result = p._replace_value_with_mode(is_train=True)
        assert result is p
        assert len(p.mode_replacements) == 0
