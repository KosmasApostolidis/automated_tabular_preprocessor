"""
Example: Basic usage of the Tabular Preprocessor

This script demonstrates how to use the TabularDataPreprocessor and
AugmentedDataPreprocessor classes for preprocessing tabular data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import from the package
from tabular_preprocessor import TabularDataPreprocessor, AugmentedDataPreprocessor


def create_sample_dataset(n_samples=1000, n_features=20, n_informative=10):
    """Create a synthetic dataset for demonstration."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    # Add some categorical columns
    df["category_A"] = np.random.choice(["low", "medium", "high"], size=n_samples)
    df["category_B"] = np.random.choice(["type_1", "type_2", "type_3", "type_4"], size=n_samples)
    
    # Introduce some missing values
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)
    df["target"] = y  # Restore target (no missing values)
    
    # Add an ID column (to be dropped)
    df["patient_id"] = [f"P{i:04d}" for i in range(n_samples)]
    
    return df


def example_basic_preprocessing():
    """Example: Basic preprocessing pipeline."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Preprocessing")
    print("=" * 60)
    
    # Create dataset
    df = create_sample_dataset()
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Split into train/test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    
    # Initialize preprocessor
    preprocessor = TabularDataPreprocessor(
        df=df_train.copy(),
        target_column="target",
        cols_to_drop=["patient_id"],
        numberical_scaler="standard",
        categorical_encoder="onehot",
        number_of_top_k_features=10,
        feature_selection_method="trees",
        missing_values_threshold=0.4,
        random_state=42
    )
    
    # Preprocess training data
    preprocessor._run_preprocessing_pipeline(is_train=True)
    df_train_processed = preprocessor.df.copy()
    print(f"\nProcessed train shape: {df_train_processed.shape}")
    
    # Preprocess test data using same transformations
    preprocessor.df = df_test.copy()
    preprocessor._run_preprocessing_pipeline(is_train=False)
    df_test_processed = preprocessor.df.copy()
    print(f"Processed test shape: {df_test_processed.shape}")
    
    # Verify column alignment
    print(f"\nTrain columns match test columns: {list(df_train_processed.columns) == list(df_test_processed.columns)}")
    
    return df_train_processed, df_test_processed


def example_with_augmentation():
    """Example: Preprocessing with SMOTE augmentation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Preprocessing with SMOTE Augmentation")
    print("=" * 60)
    
    # Create imbalanced dataset
    df = create_sample_dataset(n_samples=500)
    print(f"Original class distribution:\n{df['target'].value_counts()}")
    
    # Initialize augmented preprocessor
    preprocessor = AugmentedDataPreprocessor(
        df=df.copy(),
        target_column="target",
        cols_to_drop=["patient_id"],
        strategy="smote",
        numberical_scaler="standard",
        categorical_encoder="onehot",
        random_state=42
    )
    
    # Run pipeline with augmentation
    preprocessor._run_preprocessing_pipeline(is_train=True)
    df_augmented = preprocessor.df
    
    print(f"\nAugmented dataset shape: {df_augmented.shape}")
    print(f"Augmented class distribution:\n{df_augmented['target'].value_counts()}")
    
    return df_augmented


def example_feature_selection():
    """Example: Different feature selection methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Feature Selection Methods")
    print("=" * 60)
    
    df = create_sample_dataset(n_samples=500)
    
    # Tree-based feature selection
    preprocessor_trees = TabularDataPreprocessor(
        df=df.copy(),
        target_column="target",
        cols_to_drop=["patient_id"],
        number_of_top_k_features=5,
        feature_selection_method="trees",
        encode_categorical=True,
        random_state=42
    )
    preprocessor_trees._run_preprocessing_pipeline(is_train=True)
    print(f"\nTree-based top features: {preprocessor_trees.columns_to_keep}")
    
    # F-statistic feature selection
    preprocessor_fstat = TabularDataPreprocessor(
        df=df.copy(),
        target_column="target",
        cols_to_drop=["patient_id"],
        number_of_top_k_features=5,
        feature_selection_method="f_classif",
        encode_categorical=True,
        random_state=42
    )
    preprocessor_fstat._run_preprocessing_pipeline(is_train=True)
    print(f"F-statistic top features: {preprocessor_fstat.columns_to_keep}")


def example_outlier_removal():
    """Example: Outlier removal with IQR."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Outlier Removal")
    print("=" * 60)
    
    df = create_sample_dataset(n_samples=500)
    
    # Add some obvious outliers
    df.loc[0, "feature_0"] = 1000
    df.loc[1, "feature_1"] = -500
    
    print(f"Dataset shape before outlier removal: {df.shape}")
    
    preprocessor = TabularDataPreprocessor(
        df=df.copy(),
        target_column="target",
        cols_to_drop=["patient_id"],
        remove_outliers=True,
        iqr_multiplier=1.5,
        encode_categorical=True,
        random_state=42
    )
    
    preprocessor._run_preprocessing_pipeline(is_train=True)
    print(f"Dataset shape after outlier removal: {preprocessor.df.shape}")


if __name__ == "__main__":
    # Run all examples
    example_basic_preprocessing()
    example_with_augmentation()
    example_feature_selection()
    example_outlier_removal()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
