import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import AugmentedDataPreprocessor


class TestSMOTEAugmentation:
    def test_smote_increases_minority(self, imbalanced_df):
        p = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            encode_categorical=False,
        )
        original_minority = (imbalanced_df['target'] == 1).sum()
        p._augment_data()
        new_minority = (p.df['target'] == 1).sum()
        assert new_minority > original_minority

    def test_smote_uses_instance_random_state(self, imbalanced_df):
        p1 = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            random_state=42,
            encode_categorical=False,
        )
        p1._augment_data()

        p2 = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            random_state=42,
            encode_categorical=False,
        )
        p2._augment_data()

        pd.testing.assert_frame_equal(p1.df, p2.df)

    def test_different_seed_different_result(self, imbalanced_df):
        p1 = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            random_state=42,
            encode_categorical=False,
        )
        p1._augment_data()

        p2 = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            random_state=99,
            encode_categorical=False,
        )
        p2._augment_data()

        assert not p1.df.equals(p2.df)

    def test_augmentation_only_on_train(self, imbalanced_df):
        p = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            encode_categorical=False,
        )
        original_len = len(p.df)
        p.run_preprocessing_pipeline(is_train=False)
        assert len(p.df) == original_len


class TestAugmentedPipeline:
    def test_full_pipeline_train(self, imbalanced_df):
        p = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            numerical_scaler='standard',
            encode_categorical=False,
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert p.df.isnull().sum().sum() == 0
        assert len(p.df) > len(imbalanced_df)

    def test_pipeline_train_test_roundtrip(self, imbalanced_df):
        np.random.seed(99)
        test = pd.DataFrame({
            'f1': np.random.randn(20),
            'f2': np.random.randn(20),
            'target': np.random.choice([0, 1], 20),
        })
        p = AugmentedDataPreprocessor(
            df=imbalanced_df.copy(),
            target_column='target',
            strategy='smote',
            numerical_scaler='standard',
            encode_categorical=False,
        )
        p.run_preprocessing_pipeline(is_train=True)
        train_cols = set(p.df.columns)

        p.df = test.copy()
        p.run_preprocessing_pipeline(is_train=False)
        assert set(p.df.columns) == train_cols
