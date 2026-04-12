import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestEdgeCases:
    def test_single_feature(self):
        df = pd.DataFrame({
            'f1': np.random.randn(50),
            'target': np.random.choice([0, 1], 50),
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert 'f1' in p.df.columns
        assert 'target' in p.df.columns

    def test_all_categorical_features(self):
        df = pd.DataFrame({
            'cat1': np.random.choice(['a', 'b'], 50),
            'cat2': np.random.choice(['x', 'y', 'z'], 50),
            'target': np.random.choice(['pos', 'neg'], 50),
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
            categorical_encoder='onehot',
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert p.df.isnull().sum().sum() == 0

    def test_empty_cols_to_drop(self):
        df = pd.DataFrame({
            'f1': [1, 2, 3],
            'target': [0, 1, 0],
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
            cols_to_drop=[],
        )
        p._drop_columns()
        assert 'f1' in p.df.columns

    def test_drop_nonexistent_column(self):
        df = pd.DataFrame({
            'f1': [1, 2, 3],
            'target': [0, 1, 0],
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
            cols_to_drop=['nonexistent'],
        )
        p._drop_columns()
        assert 'f1' in p.df.columns

    def test_no_missing_values(self):
        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0],
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
        )
        p._handle_missing_values(is_train=True)
        assert p.df.isnull().sum().sum() == 0
        assert len(p.imputation_values) == 0

    def test_multiclass_target(self):
        df = pd.DataFrame({
            'f1': np.random.randn(60),
            'f2': np.random.randn(60),
            'target': np.random.choice(['A', 'B', 'C', 'D'], 60),
        })
        p = TabularDataPreprocessor(
            df=df.copy(),
            target_column='target',
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert set(p.df['target'].unique()).issubset({0, 1, 2, 3})

    def test_full_pipeline_all_features_enabled(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            numerical_scaler='standard',
            categorical_encoder='onehot',
            number_of_top_k_features=2,
            feature_selection_method='trees',
            remove_outliers=True,
            remove_highly_correlated=True,
            remove_low_variance=True,
        )
        p.run_preprocessing_pipeline(is_train=True)
        train_cols = set(p.df.columns)
        assert p.df.isnull().sum().sum() == 0

        p.df = basic_test_df.copy()
        p.run_preprocessing_pipeline(is_train=False)
        assert set(p.df.columns) == train_cols
        assert p.df.isnull().sum().sum() == 0

    def test_class_balance_check_series(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        result = p._check_class_balance(basic_train_df['target'], title="test")
        assert result is p

    def test_class_balance_check_dataframe(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        result = p._check_class_balance(basic_train_df, title="test")
        assert result is p

    def test_class_balance_check_invalid_type(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        with pytest.raises(TypeError):
            p._check_class_balance([1, 2, 3])

    def test_neural_network_with_feature_selection(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            model_type='neural_network',
            categorical_encoder='onehot',
            number_of_top_k_features=2,
            feature_selection_method='trees',
            numerical_scaler='standard',
        )
        p.run_preprocessing_pipeline(is_train=True)
        train_cols = set(p.df.columns)

        for col in p.final_target_cols:
            assert col in p.df.columns

        p.df = basic_test_df.copy()
        p.run_preprocessing_pipeline(is_train=False)
        assert set(p.df.columns) == train_cols

    def test_neural_network_target_not_scaled(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            model_type='neural_network',
            categorical_encoder='onehot',
            numerical_scaler='standard',
        )
        p.run_preprocessing_pipeline(is_train=True)
        for col in p.final_target_cols:
            assert set(p.df[col].unique()).issubset({0, 1})

    def test_neural_network_full_pipeline(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            model_type='neural_network',
            categorical_encoder='onehot',
            numerical_scaler='standard',
            number_of_top_k_features=2,
            remove_outliers=True,
            remove_highly_correlated=True,
            remove_low_variance=True,
        )
        p.run_preprocessing_pipeline(is_train=True)
        train_cols = set(p.df.columns)
        assert p.df.isnull().sum().sum() == 0

        p.df = basic_test_df.copy()
        p.run_preprocessing_pipeline(is_train=False)
        assert set(p.df.columns) == train_cols
