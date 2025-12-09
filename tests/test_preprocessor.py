"""
Unit tests for the Tabular Preprocessor library.

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from tabular_preprocessor import (
    TabularDataPreprocessor,
    AugmentedDataPreprocessor,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 200
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["target"] = y
    df["category"] = np.random.choice(["A", "B", "C"], size=n_samples)
    df["id"] = range(n_samples)
    
    return df


@pytest.fixture
def dataframe_with_missing(sample_dataframe):
    """Create a DataFrame with missing values."""
    df = sample_dataframe.copy()
    # Introduce missing values
    df.loc[0:10, "feat_0"] = np.nan
    df.loc[5:15, "feat_1"] = np.nan
    df.loc[0:5, "category"] = np.nan
    return df


class TestTabularDataPreprocessor:
    """Tests for TabularDataPreprocessor class."""
    
    def test_initialization(self, sample_dataframe):
        """Test preprocessor initialization."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe,
            target_column="target"
        )
        assert preprocessor.df is not None
        assert preprocessor.target_column == "target"
        assert preprocessor.random_state == 42
    
    def test_drop_columns(self, sample_dataframe):
        """Test column dropping functionality."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            cols_to_drop=["id"]
        )
        preprocessor._drop_columns()
        assert "id" not in preprocessor.df.columns
    
    def test_drop_duplicates(self, sample_dataframe):
        """Test duplicate removal."""
        df = pd.concat([sample_dataframe, sample_dataframe.iloc[:5]], ignore_index=True)
        preprocessor = TabularDataPreprocessor(df=df, target_column="target")
        
        initial_len = len(preprocessor.df)
        preprocessor._drop_duplicates()
        assert len(preprocessor.df) < initial_len
    
    def test_handle_missing_values(self, dataframe_with_missing):
        """Test missing value handling."""
        preprocessor = TabularDataPreprocessor(
            df=dataframe_with_missing.copy(),
            target_column="target"
        )
        
        assert preprocessor.df.isnull().sum().sum() > 0
        preprocessor._handle_missing_values(is_train=True)
        assert preprocessor.df.isnull().sum().sum() == 0
    
    def test_categorical_encoding_onehot(self, sample_dataframe):
        """Test one-hot encoding of categorical features."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            categorical_encoder="onehot"
        )
        preprocessor._encode_categorical_features(is_train=True)
        
        # Original 'category' column should be replaced by dummies
        assert "category" not in preprocessor.df.columns
        assert any("category_" in col for col in preprocessor.df.columns)
    
    def test_categorical_encoding_factorize(self, sample_dataframe):
        """Test factorize encoding of categorical features."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            categorical_encoder="factorize"
        )
        preprocessor._encode_categorical_features(is_train=True)
        
        # Category column should now be numeric
        assert preprocessor.df["category"].dtype in [np.int64, np.int32, int]
    
    def test_scaling_standard(self, sample_dataframe):
        """Test standard scaling."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            cols_to_drop=["id", "category"],
            numberical_scaler="standard"
        )
        preprocessor._drop_columns()
        preprocessor._scale_numerical_features(is_train=True)
        
        # Check that features are approximately standardized
        numeric_cols = preprocessor.df.select_dtypes(include=np.number).columns
        numeric_cols = [c for c in numeric_cols if c != "target"]
        
        for col in numeric_cols:
            assert abs(preprocessor.df[col].mean()) < 0.1
            assert abs(preprocessor.df[col].std() - 1.0) < 0.1
    
    def test_feature_selection_trees(self, sample_dataframe):
        """Test tree-based feature selection."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            cols_to_drop=["id", "category"],
            number_of_top_k_features=5,
            feature_selection_method="trees"
        )
        preprocessor._drop_columns()
        preprocessor._select_top_k_features(is_train=True)
        
        # Should have 5 features + target
        assert preprocessor.df.shape[1] == 6
        assert "target" in preprocessor.df.columns
    
    def test_feature_selection_fclassif(self, sample_dataframe):
        """Test F-statistic feature selection."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            cols_to_drop=["id", "category"],
            number_of_top_k_features=5,
            feature_selection_method="f_classif"
        )
        preprocessor._drop_columns()
        preprocessor._select_top_k_features(is_train=True)
        
        # Should have 5 features + target
        assert preprocessor.df.shape[1] == 6
    
    def test_full_pipeline(self, sample_dataframe):
        """Test the complete preprocessing pipeline."""
        preprocessor = TabularDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            cols_to_drop=["id"],
            numberical_scaler="standard",
            categorical_encoder="onehot",
            number_of_top_k_features=5
        )
        
        preprocessor._run_preprocessing_pipeline(is_train=True)
        
        assert preprocessor.df is not None
        assert "target" in preprocessor.df.columns
        assert preprocessor.df.isnull().sum().sum() == 0
    
    def test_train_test_consistency(self, sample_dataframe):
        """Test that train and test preprocessing produces consistent columns."""
        df_train = sample_dataframe.iloc[:150].copy()
        df_test = sample_dataframe.iloc[150:].copy()
        
        preprocessor = TabularDataPreprocessor(
            df=df_train,
            target_column="target",
            cols_to_drop=["id"],
            categorical_encoder="onehot"
        )
        
        # Process train
        preprocessor._run_preprocessing_pipeline(is_train=True)
        train_cols = set(preprocessor.df.columns)
        
        # Process test
        preprocessor.df = df_test
        preprocessor._run_preprocessing_pipeline(is_train=False)
        test_cols = set(preprocessor.df.columns)
        
        assert train_cols == test_cols


class TestAugmentedDataPreprocessor:
    """Tests for AugmentedDataPreprocessor class."""
    
    def test_smote_augmentation(self, sample_dataframe):
        """Test SMOTE augmentation."""
        # Create imbalanced dataset
        df = sample_dataframe.copy()
        df = df[df["target"] == 0].head(150)
        df = pd.concat([df, sample_dataframe[sample_dataframe["target"] == 1].head(50)])
        
        preprocessor = AugmentedDataPreprocessor(
            df=df.copy(),
            target_column="target",
            cols_to_drop=["id"],
            strategy="smote",
            categorical_encoder="onehot"
        )
        
        initial_counts = df["target"].value_counts()
        preprocessor._run_preprocessing_pipeline(is_train=True)
        final_counts = preprocessor.df["target"].value_counts()
        
        # SMOTE should balance the classes
        assert final_counts[0] == final_counts[1]
    
    def test_augmentation_only_on_train(self, sample_dataframe):
        """Test that augmentation only runs on training data."""
        preprocessor = AugmentedDataPreprocessor(
            df=sample_dataframe.copy(),
            target_column="target",
            cols_to_drop=["id"],
            strategy="smote",
            categorical_encoder="onehot"
        )
        
        # Process as test data
        initial_len = len(preprocessor.df)
        preprocessor._run_preprocessing_pipeline(is_train=False)
        
        # Length should not increase for test data
        assert len(preprocessor.df) <= initial_len


class TestOutlierRemoval:
    """Tests for outlier removal functionality."""
    
    def test_iqr_outlier_removal(self, sample_dataframe):
        """Test IQR-based outlier removal."""
        df = sample_dataframe.copy()
        # Add obvious outliers
        df.loc[0, "feat_0"] = 1000
        df.loc[1, "feat_1"] = -1000
        
        preprocessor = TabularDataPreprocessor(
            df=df,
            target_column="target",
            cols_to_drop=["id", "category"],
            remove_outliers=True,
            iqr_multiplier=1.5
        )
        
        initial_len = len(preprocessor.df)
        preprocessor._drop_columns()
        preprocessor._remove_outliers_iqr()
        
        assert len(preprocessor.df) < initial_len


class TestCorrelationRemoval:
    """Tests for highly correlated feature removal."""
    
    def test_remove_highly_correlated(self, sample_dataframe):
        """Test removal of highly correlated features."""
        df = sample_dataframe.copy()
        # Add a perfectly correlated column
        df["feat_copy"] = df["feat_0"] * 1.0
        
        preprocessor = TabularDataPreprocessor(
            df=df,
            target_column="target",
            cols_to_drop=["id", "category"],
            remove_highly_correlated=True
        )
        
        preprocessor._drop_columns()
        preprocessor._remove_highly_correlated(threshold=0.8)
        
        # One of the correlated columns should be removed
        assert "feat_copy" not in preprocessor.df.columns or "feat_0" not in preprocessor.df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
