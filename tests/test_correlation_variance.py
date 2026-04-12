import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestHighlyCorrelated:
    def test_drops_correlated_features(self, correlated_df):
        p = TabularDataPreprocessor(
            df=correlated_df.copy(),
            target_column='target',
            remove_highly_correlated=True,
        )
        p._remove_highly_correlated_features(threshold=0.8, is_train=True)
        assert p.df.shape[1] < correlated_df.shape[1]
        assert len(p.highly_correlated_cols) > 0

    def test_target_not_dropped(self, correlated_df):
        p = TabularDataPreprocessor(
            df=correlated_df.copy(),
            target_column='target',
            remove_highly_correlated=True,
        )
        p._remove_highly_correlated_features(is_train=True)
        assert 'target' in p.df.columns

    def test_train_test_consistency(self, correlated_df):
        np.random.seed(99)
        n = 30
        base = np.random.randn(n)
        test = pd.DataFrame({
            'f1': base,
            'f2': base + np.random.randn(n) * 0.01,
            'f3': np.random.randn(n) * 10,
            'target': np.random.choice([0, 1], n),
        })
        p = TabularDataPreprocessor(
            df=correlated_df.copy(),
            target_column='target',
            remove_highly_correlated=True,
        )
        p._remove_highly_correlated_features(is_train=True)
        train_cols = set(p.df.columns)

        p.df = test.copy()
        p._remove_highly_correlated_features(is_train=False)
        assert set(p.df.columns) == train_cols

    def test_wired_into_pipeline(self, correlated_df):
        p = TabularDataPreprocessor(
            df=correlated_df.copy(),
            target_column='target',
            remove_highly_correlated=True,
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert len(p.highly_correlated_cols) > 0

    def test_disabled_by_default(self, correlated_df):
        p = TabularDataPreprocessor(
            df=correlated_df.copy(),
            target_column='target',
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert len(p.highly_correlated_cols) == 0


class TestLowVariance:
    def test_drops_constant_columns(self, low_variance_df):
        p = TabularDataPreprocessor(
            df=low_variance_df.copy(),
            target_column='target',
            remove_low_variance=True,
        )
        p._remove_low_variance_features(threshold=0.01, is_train=True)
        assert 'constant' not in p.df.columns
        assert 'near_constant' not in p.df.columns
        assert 'varied' in p.df.columns

    def test_target_not_dropped(self, low_variance_df):
        p = TabularDataPreprocessor(
            df=low_variance_df.copy(),
            target_column='target',
            remove_low_variance=True,
        )
        p._remove_low_variance_features(is_train=True)
        assert 'target' in p.df.columns

    def test_train_test_consistency(self, low_variance_df):
        test = pd.DataFrame({
            'constant': [5.0] * 20,
            'near_constant': [1.0] * 20,
            'varied': np.random.randn(20),
            'target': np.random.choice([0, 1], 20),
        })
        p = TabularDataPreprocessor(
            df=low_variance_df.copy(),
            target_column='target',
            remove_low_variance=True,
        )
        p._remove_low_variance_features(is_train=True)
        train_cols = set(p.df.columns)

        p.df = test.copy()
        p._remove_low_variance_features(is_train=False)
        assert set(p.df.columns) == train_cols

    def test_wired_into_pipeline(self, low_variance_df):
        p = TabularDataPreprocessor(
            df=low_variance_df.copy(),
            target_column='target',
            remove_low_variance=True,
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert 'constant' not in p.df.columns

    def test_disabled_by_default(self, low_variance_df):
        p = TabularDataPreprocessor(
            df=low_variance_df.copy(),
            target_column='target',
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert 'constant' in p.df.columns
