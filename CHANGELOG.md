# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-XX-XX

### Added
- Initial release of the Tabular Preprocessor library
- `AbstractPreprocessor` base class with core preprocessing methods
- `TabularDataPreprocessor` class with:
  - Missing value handling (threshold-based dropping, median/mean/mode imputation)
  - Duplicate row removal
  - Column dropping
  - Placeholder value replacement with mode
  - One-hot and factorize categorical encoding
  - Target column encoding (label or one-hot for neural networks)
  - IQR-based outlier removal
  - Highly correlated feature removal
  - Low variance feature removal
  - Tree-based and F-statistic feature selection
  - StandardScaler, MinMaxScaler, and RobustScaler support
- `AugmentedDataPreprocessor` class with:
  - SMOTE oversampling
  - CTGAN synthetic data generation
  - TVAE synthetic data generation
- Train/test consistency (learned parameters on train, applied to test)
- Comprehensive documentation and examples
- Unit test suite

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- imbalanced-learn >= 0.9.0
- ctgan >= 0.7.0
