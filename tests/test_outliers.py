import pandas as pd
import numpy as np
import pytest
from automated_tabular_preprocessor import TabularDataPreprocessor


class TestOutlierRemoval:
    def test_removes_extreme_outliers(self, df_with_outliers):
        p = TabularDataPreprocessor(
            df=df_with_outliers.copy(),
            target_column='target',
            remove_outliers=True,
        )
        p._remove_outliers_iqr()
        assert len(p.df) < len(df_with_outliers)

    def test_custom_iqr_multiplier(self, df_with_outliers):
        p_strict = TabularDataPreprocessor(
            df=df_with_outliers.copy(),
            target_column='target',
            iqr_multiplier=1.0,
        )
        p_strict._remove_outliers_iqr()

        p_loose = TabularDataPreprocessor(
            df=df_with_outliers.copy(),
            target_column='target',
            iqr_multiplier=3.0,
        )
        p_loose._remove_outliers_iqr()

        assert len(p_strict.df) <= len(p_loose.df)

    def test_target_not_used_for_outlier_detection(self, df_with_outliers):
        p = TabularDataPreprocessor(
            df=df_with_outliers.copy(),
            target_column='target',
        )
        p._remove_outliers_iqr()
        assert 'target' in p.df.columns

    def test_outliers_only_removed_on_train(self, df_with_outliers):
        p = TabularDataPreprocessor(
            df=df_with_outliers.copy(),
            target_column='target',
            remove_outliers=True,
        )
        original_len = len(p.df)
        p.run_preprocessing_pipeline(is_train=False)
        assert len(p.df) == original_len

    def test_pipeline_wires_outlier_flag(self, df_with_outliers):
        p = TabularDataPreprocessor(
            df=df_with_outliers.copy(),
            target_column='target',
            remove_outliers=True,
        )
        p.run_preprocessing_pipeline(is_train=True)
        assert len(p.df) < len(df_with_outliers)

    def test_no_outlier_removal_when_disabled(self, df_with_outliers):
        p = TabularDataPreprocessor(
            df=df_with_outliers.copy(),
            target_column='target',
            remove_outliers=False,
        )
        original_len = len(p.df)
        p.run_preprocessing_pipeline(is_train=True)
        assert len(p.df) == original_len
