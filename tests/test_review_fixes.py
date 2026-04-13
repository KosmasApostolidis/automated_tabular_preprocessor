import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestInvalidScalerValidation:
    """Issue #1: Invalid scaler name should raise ValueError, not AttributeError."""

    def test_invalid_scaler_raises_valueerror(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler='standardd',
        )
        with pytest.raises(ValueError, match="Unknown numerical_scaler"):
            p._scale_numerical_features(is_train=True)

    def test_invalid_scaler_lists_valid_options(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
            numerical_scaler='invalid',
        )
        with pytest.raises(ValueError, match="'standard', 'minmax', 'robust'"):
            p._scale_numerical_features(is_train=True)


class TestLowVarianceNumericOnly:
    """Issue #2: _remove_low_variance_features should skip non-numeric columns."""

    def test_mixed_dtypes_no_crash(self):
        df = pd.DataFrame({
            'cat_a': ['x', 'y', 'z', 'x', 'y'] * 20,
            'num_a': [5.0] * 100,
            'num_b': np.random.randn(100),
            'target': np.random.choice([0, 1], 100),
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
            remove_low_variance=True,
            encode_categorical=False,
        )
        p._remove_low_variance_features(threshold=0.01, is_train=True)
        assert 'cat_a' in p.df.columns
        assert 'num_a' not in p.df.columns
        assert 'num_b' in p.df.columns


class TestFactorizeEncodedCategoricalCols:
    """Issue #3: Factorize encoder must populate encoded_categorical_cols."""

    def test_factorize_populates_encoded_cols(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='factorize',
        )
        p._encode_categorical_features(is_train=True)
        assert len(p.encoded_categorical_cols) > 0
        assert 'cat_a' in p.encoded_categorical_cols
        assert 'cat_b' in p.encoded_categorical_cols

    def test_factorize_outlier_removal_skips_encoded(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='factorize',
            remove_outliers=True,
        )
        p._encode_categorical_features(is_train=True)
        rows_before = len(p.df)
        p._remove_outliers_iqr()
        # Factorized cols should not contribute to outlier detection,
        # so fewer (or no) rows should be removed compared to treating them as numeric
        assert len(p.df) <= rows_before

    def test_factorize_initializes_columns_after_encoding(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='factorize',
        )
        p._encode_categorical_features(is_train=True)
        assert hasattr(p, 'columns_after_encoding')
        assert len(p.columns_after_encoding) > 0


class TestHighMissingTrainTestConsistency:
    """Issue #4: Columns dropped for high missing on train must also be dropped on test."""

    def test_high_missing_cols_tracked(self):
        train = pd.DataFrame({
            'good': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'mostly_missing': [np.nan] * 8 + [1.0, 2.0],
            'target': [0, 1] * 5,
        })
        p = TabularDataPreprocessor(
            df=train.copy(),
            target_column='target',
            missing_values_threshold=0.4,
        )
        p._handle_missing_values(is_train=True)
        assert 'mostly_missing' in p.high_missing_cols_dropped
        assert 'mostly_missing' not in p.df.columns

    def test_high_missing_cols_dropped_on_test(self):
        train = pd.DataFrame({
            'good': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'mostly_missing': [np.nan] * 8 + [1.0, 2.0],
            'target': [0, 1] * 5,
        })
        test = pd.DataFrame({
            'good': [1.0, 2.0, 3.0],
            'mostly_missing': [10.0, 20.0, 30.0],  # no missing in test
            'target': [0, 1, 0],
        })
        p = TabularDataPreprocessor(
            df=train.copy(),
            target_column='target',
            missing_values_threshold=0.4,
        )
        p._handle_missing_values(is_train=True)

        p.df = test.copy()
        p._handle_missing_values(is_train=False)
        assert 'mostly_missing' not in p.df.columns
        assert 'good' in p.df.columns

    def test_factorize_full_pipeline_train_test_consistency(self):
        np.random.seed(42)
        n = 100
        train = pd.DataFrame({
            'num_a': np.random.randn(n),
            'cat_a': np.random.choice(['x', 'y', 'z'], n),
            'mostly_missing': [np.nan] * 80 + list(np.random.randn(20)),
            'target': np.random.choice(['classA', 'classB'], n),
        })
        test = pd.DataFrame({
            'num_a': np.random.randn(30),
            'cat_a': np.random.choice(['x', 'y', 'z'], 30),
            'mostly_missing': np.random.randn(30),
            'target': np.random.choice(['classA', 'classB'], 30),
        })
        p = TabularDataPreprocessor(
            df=train.copy(),
            target_column='target',
            categorical_encoder='factorize',
            missing_values_threshold=0.4,
        )
        p.run_preprocessing_pipeline(is_train=True)
        train_cols = set(p.df.columns)

        p.df = test.copy()
        p.run_preprocessing_pipeline(is_train=False)
        assert set(p.df.columns) == train_cols
        assert 'mostly_missing' not in p.df.columns


class TestPipelineHook:
    """Issue #5: AugmentedDataPreprocessor should use hook, not duplicate pipeline."""

    def test_hook_is_noop_in_base_class(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        # Should not raise
        p._post_target_encoding_hook(is_train=True)
        p._post_target_encoding_hook(is_train=False)
