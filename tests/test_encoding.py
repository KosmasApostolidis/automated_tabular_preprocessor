import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestCategoricalEncoding:
    def test_onehot_creates_binary_columns(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='onehot',
        )
        p._encode_categorical_features(is_train=True)

        for col in p.encoded_categorical_cols:
            assert set(p.df[col].unique()).issubset({0, 1})

    def test_onehot_train_test_alignment(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='onehot',
        )
        p._encode_categorical_features(is_train=True)
        train_cols = set(p.df.columns)

        p.df = basic_test_df.copy()
        p._encode_categorical_features(is_train=False)
        test_cols = set(p.df.columns)

        assert train_cols == test_cols

    def test_onehot_unseen_category_filled_with_zero(self):
        train = pd.DataFrame({
            'cat': ['a', 'b', 'c', 'a', 'b'],
            'target': [0, 1, 0, 1, 0],
        })
        test = pd.DataFrame({
            'cat': ['a', 'd'],
            'target': [0, 1],
        })
        p = TabularDataPreprocessor(
            df=train.copy(),
            target_column='target',
            categorical_encoder='onehot',
        )
        p._encode_categorical_features(is_train=True)
        train_cols = p.df.columns.tolist()

        p.df = test.copy()
        p._encode_categorical_features(is_train=False)
        assert set(p.df.columns) == set(train_cols)

    def test_factorize_encoding(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='factorize',
        )
        p._encode_categorical_features(is_train=True)
        assert pd.api.types.is_integer_dtype(p.df['cat_a'])
        assert 'cat_a' in p.cat_encoding_mappings

    def test_factorize_test_applies_mapping(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='factorize',
        )
        p._encode_categorical_features(is_train=True)
        mapping_a = p.cat_encoding_mappings['cat_a']

        p.df = basic_test_df.copy()
        p._encode_categorical_features(is_train=False)
        for val in p.df['cat_a'].unique():
            assert val in mapping_a.values() or val == -1

    def test_encoded_categorical_cols_tracks_only_new_columns(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='onehot',
        )
        original_numeric = set(basic_train_df.select_dtypes(include=np.number).columns)
        p._encode_categorical_features(is_train=True)

        for col in p.encoded_categorical_cols:
            assert col not in original_numeric, f"'{col}' is an original numeric, not a new one-hot column"

    def test_target_excluded_from_categorical_encoding(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            categorical_encoder='onehot',
        )
        p._encode_categorical_features(is_train=True)
        assert 'target' in p.df.columns
        assert p.df['target'].dtype == object


class TestTargetEncoding:
    def test_label_encoding_saves_mapping(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        p._encode_target_column(is_train=True)
        assert p.target_encoding_mapping is not None
        assert 'classA' in p.target_encoding_mapping
        assert 'classB' in p.target_encoding_mapping

    def test_label_encoding_consistent_train_test(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
        )
        p._encode_target_column(is_train=True)
        mapping = p.target_encoding_mapping.copy()

        p.df = basic_test_df.copy()
        p._encode_target_column(is_train=False)

        for original_val, encoded_val in mapping.items():
            mask = basic_test_df['target'] == original_val
            if mask.any():
                assert (p.df.loc[mask.values, 'target'] == encoded_val).all()

    def test_numeric_target_skips_encoding(self, numeric_only_df):
        p = TabularDataPreprocessor(
            df=numeric_only_df.copy(),
            target_column='label',
        )
        original_values = numeric_only_df['label'].copy()
        p._encode_target_column(is_train=True)
        pd.testing.assert_series_equal(p.df['label'], original_values)

    def test_neural_network_onehot_target(self, basic_train_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            model_type='neural_network',
        )
        p._encode_target_column(is_train=True)
        assert 'target' not in p.df.columns
        assert len(p.final_target_cols) > 0
        for col in p.final_target_cols:
            assert col in p.df.columns
            assert set(p.df[col].unique()).issubset({0, 1})

    def test_neural_network_target_train_test_alignment(self, basic_train_df, basic_test_df):
        p = TabularDataPreprocessor(
            df=basic_train_df.copy(),
            target_column='target',
            model_type='neural_network',
        )
        p._encode_target_column(is_train=True)
        train_target_cols = set(p.final_target_cols)

        p.df = basic_test_df.copy()
        p._encode_target_column(is_train=False)
        for col in train_target_cols:
            assert col in p.df.columns
