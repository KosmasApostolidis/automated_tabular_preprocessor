import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestBasicPipeline:
    def test_train_test_roundtrip(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            numerical_scaler='standard',
            categorical_encoder='onehot',
        )
        p.run_preprocessing_pipeline(is_train=True)
        train_cols = set(p.df.columns)

        p.df = basic_test_df.copy()
        p.run_preprocessing_pipeline(is_train=False)
        test_cols = set(p.df.columns)

        assert train_cols == test_cols

    def test_output_is_numeric(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='onehot',
        )
        p.run_preprocessing_pipeline(is_train=True)

        for col in p.df.columns:
            assert pd.api.types.is_numeric_dtype(p.df[col]), f"Column '{col}' is not numeric"

    def test_no_missing_values_after_pipeline(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert p.df.isnull().sum().sum() == 0

    def test_no_duplicates_after_pipeline(self, basic_train_df):
        df = pd.concat([basic_train_df, basic_train_df.iloc[:5]], ignore_index=True)
        p = TabularDataPreprocessor(df=df.copy(), target_column='target')
        p.run_preprocessing_pipeline(is_train=True)
        assert len(p.df) == len(p.df.drop_duplicates())

    def test_numeric_only_input(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler='standard',
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert 'label' in p.df.columns
        assert p.df.isnull().sum().sum() == 0

    def test_drop_columns(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            cols_to_drop=['num_b', 'cat_b'],
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert 'num_b' not in p.df.columns
        assert 'cat_b' not in p.df.columns

    def test_returns_self(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        result = p.run_preprocessing_pipeline(is_train=True)
        assert result is p
