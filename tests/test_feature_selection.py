import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestFeatureSelection:
    @pytest.mark.parametrize("method", ["trees", "f_classif"])
    def test_selects_top_k(self, numeric_only_df, method):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            number_of_top_k_features=2,
            feature_selection_method=method,
        )
        p._select_top_k_features(is_train=True)
        assert p.df.shape[1] == 3  # 2 features + target

    def test_target_always_kept(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            number_of_top_k_features=1,
            feature_selection_method='trees',
        )
        p._select_top_k_features(is_train=True)
        assert 'label' in p.df.columns

    def test_zero_k_skips_selection(self, numeric_only_df):
        original_cols = numeric_only_df.columns.tolist()
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            number_of_top_k_features=0,
        )
        p._select_top_k_features(is_train=True)
        assert p.df.columns.tolist() == original_cols

    def test_test_uses_train_columns(self, numeric_only_df):
        np.random.seed(99)
        test = pd.DataFrame({
            'f1': np.random.randn(20),
            'f2': np.random.randn(20),
            'f3': np.random.randn(20),
            'label': np.random.choice([0, 1, 2], 20),
        })
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            number_of_top_k_features=2,
            feature_selection_method='trees',
        )
        p._select_top_k_features(is_train=True)
        train_cols = set(p.df.columns)

        p.df = test.copy()
        p._select_top_k_features(is_train=False)
        assert set(p.df.columns) == train_cols

    def test_columns_to_keep_stored(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            number_of_top_k_features=2,
            feature_selection_method='trees',
        )
        p._select_top_k_features(is_train=True)
        assert p.columns_to_keep is not None
        assert len(p.columns_to_keep) == 3  # 2 + target
