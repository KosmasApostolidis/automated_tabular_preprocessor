import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestMissingValues:
    def test_drops_columns_above_threshold(self, df_with_missing):
        p = TabularDataPreprocessor(
            df=df_with_missing.copy(),
            target_column='target',
            missing_values_threshold=0.4,
        )
        p._drop_columns()
        p._handle_missing_values(is_train=True)
        assert 'num_b' not in p.df.columns

    def test_imputes_below_threshold(self, df_with_missing):
        p = TabularDataPreprocessor(
            df=df_with_missing.copy(),
            target_column='target',
            missing_values_threshold=0.4,
        )
        p._drop_columns()
        p._handle_missing_values(is_train=True)
        assert p.df['num_a'].isnull().sum() == 0
        assert 'num_a' in p.imputation_values

    def test_categorical_imputation(self, df_with_missing):
        p = TabularDataPreprocessor(
            df=df_with_missing.copy(),
            target_column='target',
            missing_values_threshold=0.4,
        )
        p._drop_columns()
        p._handle_missing_values(is_train=True)
        assert p.df['cat_a'].isnull().sum() == 0

    def test_test_set_uses_training_imputation(self):
        train = pd.DataFrame({
            'num': [1.0, 2.0, np.nan, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0],
        })
        test = pd.DataFrame({
            'num': [10.0, np.nan, 30.0],
            'target': [0, 1, 0],
        })
        p = TabularDataPreprocessor(
            df=train.copy(),
            target_column='target',
        )
        p._drop_columns()
        p._handle_missing_values(is_train=True)
        train_impute_val = p.imputation_values['num']

        p.df = test.copy()
        p._handle_missing_values(is_train=False)
        assert p.df['num'].iloc[1] == train_impute_val

    def test_no_missing_columns_skips_gracefully(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
        )
        p._drop_columns()
        p._handle_missing_values(is_train=True)
        assert len(p.imputation_values) == 0
