# Automated Tabular Preprocessing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Below you can find an extensive description of the automated tabular preprocessing tool with example usage.

## Features

- **Missing Value Handling**: Drops missing values above a given threshold and performs imputation (mean/median for numerical, mode for categorical) for the values below the threshold
- **Categorical Encoding**: Performs One-hot encoding and factorize (label) encoding with in train/test splits
- **Feature Scaling**: StandardScaler, MinMaxScaler, and RobustScaler
- **Feature Selection**: Tree-based importance (ExtraTreesClassifier) and F-statistic (ANOVA) methods (f_classif)
- **Outlier Removal**: Performs outlier detection and removal using IQR
- **Correlation Filtering**: Removes highly correlated features
- **Low Variance Filtering**: Remove features that have variance close to zero
- **Data Augmentation**: Performs data augmentation using SMOTE, CTGAN, and TVAE
- **Train/Test Consistency**: Applies the transformations learned from the training data to test data

## Installation

```bash
# Clone the repository
git clone https://github.com/KosmasApostolidis/automated_tabular_preprocessor.git
cd automated_tabular_preprocessor

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Basic Preprocessing

```python
import pandas as pd
from tabular_preprocessor import TabularDataPreprocessor

# Load your data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Initialize preprocessor
preprocessor = TabularDataPreprocessor(
    df                        = df_train,
    target_column             = "label",
    cols_to_drop              = ["id", "timestamp"],
    numberical_scaler         = "standard",
    categorical_encoder       = "onehot",
    number_of_top_k_features  = 10,
    feature_selection_method  = "trees",
    random_state              = 42
)

# Preprocess training data (learns parameters)
preprocessor._run_preprocessing_pipeline(is_train=True)
df_train_processed = preprocessor.df

# Preprocess test data (applies learned parameters)
preprocessor.df = df_test
preprocessor._run_preprocessing_pipeline(is_train=False)
df_test_processed = preprocessor.df
```

### With Data Augmentation (for Imbalanced Data)

```python
from tabular_preprocessor import AugmentedDataPreprocessor

# SMOTE augmentation
preprocessor = AugmentedDataPreprocessor(
    df                = df_train,
    target_column     = "label",
    strategy          = "smote",  # or "ctgan", "tvae"
    numberical_scaler = "standard",
    random_state      = 42
)

preprocessor._run_preprocessing_pipeline(is_train=True)
df_augmented = preprocessor.df
```

## Configuration Options

### TabularDataPreprocessor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | None | Input DataFrame |
| `target_column` | str | None | Name of the target variable |
| `cols_to_drop` | list | [] | Columns to exclude from preprocessing |
| `value_to_replace` | any | None | Placeholder value to replace with mode |
| `missing_values_threshold` | float | 0.4 | Drop columns with missing ratio above this |
| `random_state` | int | 42 | Random seed for reproducibility |
| `numberical_scaler` | str | "standard" | Scaler type: "standard", "minmax", "robust" |
| `categorical_encoder` | str | "onehot" | Encoder: "onehot", "factorize" |
| `number_of_top_k_features` | int | 0 | Number of features to select (0 = all) |
| `feature_selection_method` | str | "trees" | Method: "trees", "f_classif" |
| `features_to_skip_scaling` | list | [] | Columns to exclude from scaling |
| `model_type` | str | "classical" | "classical" or "neural_network" (affects target encoding) |
| `iqr_multiplier` | float | 1.5 | IQR multiplier for outlier detection |
| `remove_outliers` | bool | False | Enable outlier removal |
| `remove_highly_correlated` | bool | False | Remove features with correlation > 0.8 |
| `remove_low_variance` | bool | False | Remove near-zero variance features |
| `encode_categorical` | bool | True | Enable categorical encoding |

### AugmentedDataPreprocessor

Inherits all parameters from `TabularDataPreprocessor`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | str | "smote" | The dataset augmentation strategy to be performed: "smote", "ctgan" or "tvae" |

## Pipeline Steps

The preprocessing pipeline executes in the following order:

1. **Drop User-Specified Columns** - Remove unnecessary columns from the dataset
2. **Drop Duplicates** - Remove duplicate rows
3. **Handle Missing Values** - Drops missing columns above a given threshold and imputes the remaining
4. **Replace Placeholder Values** - Replace special values with mode
5. **Remove Outliers** (train only) - Performs outlier removal based on the IQR method
6. **Encode Categorical Features** - One-hot or label encoding
7. **Encode Target Column** - Label or one-hot (for neural networks)
8. **Data Augmentation** (train only, AugmentedDataPreprocessor) - SMOTE/CTGAN/TVAE
9. **Feature Selection** - Select top-K features
10. **Scale Numerical Features** - Apply chosen scaler

## Advanced Usage

### Custom Feature Selection

```python
# Tree-based feature importance
preprocessor = TabularDataPreprocessor(
    df                        = df_train,
    target_column             = "label",
    number_of_top_k_features  = 15,
    feature_selection_method  = "trees"
)

# ANOVA F-statistic feature importance
preprocessor = TabularDataPreprocessor(
    df                        = df_train,
    target_column             = "label",
    number_of_top_k_features  = 15,
    feature_selection_method  = "f_classif"
)
```

### Neural Network Target Encoding

```python
# One-hot encode target for multi-class neural networks
preprocessor = TabularDataPreprocessor(
    df            = df_train,
    target_column = "label",
    model_type    = "neural_network"  # Target will be one-hot encoded
)
```

### Accessing Learned Parameters

```python
# After fitting on training data
preprocessor._run_preprocessing_pipeline(is_train=True)

# Access learned transformations
print(preprocessor.scaler_object)           # Fitted scaler
print(preprocessor.imputation_values)       # Imputation statistics
print(preprocessor.columns_to_keep)         # Selected feature names
print(preprocessor.cat_encoding_mappings)   # Categorical encoding maps
```

## Requirements

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- imbalanced-learn >= 0.9.0
- ctgan >= 0.7.0

## Project Structure

```
tabular-preprocessor/
├── tabular_preprocessor/
│   ├── __init__.py
│   └── preprocessor.py
├── tests/
│   ├── __init__.py
│   └── test_preprocessor.py
├── examples/
│   └── basic_usage.py
├── docs/
├── requirements.txt
├── setup.py
├── LICENSE
├── CHANGELOG.md
└── README.md
```

## Contributing

Contributions are available! You can submit your contribution by making a Pull Request.

1. Fork the repository
2. Create your branch named "test_branch" (`git checkout -b test_branch/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin test_branch/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{tabular_preprocessor,
  author = {Your Name},
  title = {Automated Tabular Preprocessing},
  year = {2025},
  url = {https://github.com/yourusername/tabular-preprocessor}
}
```
