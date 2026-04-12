import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestScaling:
    @pytest.mark.parametrize("scaler", ["standard", "minmax", "robust"])
    def test_scaler_types(self, numeric_only_df, scaler):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler=scaler,
        )
        p._scale_numerical_features(is_train=True)
        assert p.scaler_object is not None
        assert p.df['f1'].std() != numeric_only_df['f1'].std() or scaler == 'robust'

    def test_standard_scaler_mean_zero(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler='standard',
        )
        p._scale_numerical_features(is_train=True)
        for col in ['f1', 'f2', 'f3']:
            assert abs(p.df[col].mean()) < 1e-10

    def test_minmax_scaler_range(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler='minmax',
        )
        p._scale_numerical_features(is_train=True)
        for col in ['f1', 'f2', 'f3']:
            assert p.df[col].min() >= -1e-10
            assert p.df[col].max() <= 1.0 + 1e-10

    def test_target_not_scaled(self, numeric_only_df):
        original_target = numeric_only_df['label'].copy()
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler='standard',
        )
        p._scale_numerical_features(is_train=True)
        pd.testing.assert_series_equal(p.df['label'], original_target)

    def test_test_uses_train_scaler(self):
        np.random.seed(42)
        train = pd.DataFrame({
            'f1': np.random.randn(50),
            'target': np.random.choice([0, 1], 50),
        })
        test = pd.DataFrame({
            'f1': np.random.randn(20) + 5,
            'target': np.random.choice([0, 1], 20),
        })
        p = TabularDataPreprocessor(
            df=train.copy(),
            target_column='target',
            numerical_scaler='standard',
        )
        p._scale_numerical_features(is_train=True)

        p.df = test.copy()
        p._scale_numerical_features(is_train=False)
        assert p.df['f1'].mean() > 1.0

    def test_skip_scaling_columns(self, numeric_only_df):
        original_f2 = numeric_only_df['f2'].copy()
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler='standard',
            features_to_skip_scaling=['f2'],
        )
        p._scale_numerical_features(is_train=True)
        pd.testing.assert_series_equal(p.df['f2'], original_f2)
        assert abs(p.df['f1'].mean()) < 1e-10
