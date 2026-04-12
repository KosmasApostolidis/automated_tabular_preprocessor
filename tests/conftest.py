import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def basic_train_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'num_a': np.random.randn(n),
        'num_b': np.random.randn(n) * 10 + 5,
        'cat_a': np.random.choice(['x', 'y', 'z'], n),
        'cat_b': np.random.choice(['m', 'n'], n),
        'target': np.random.choice(['classA', 'classB'], n),
    })


@pytest.fixture
def basic_test_df():
    np.random.seed(99)
    n = 30
    return pd.DataFrame({
        'num_a': np.random.randn(n),
        'num_b': np.random.randn(n) * 10 + 5,
        'cat_a': np.random.choice(['x', 'y', 'z'], n),
        'cat_b': np.random.choice(['m', 'n'], n),
        'target': np.random.choice(['classA', 'classB'], n),
    })


@pytest.fixture
def numeric_only_df():
    np.random.seed(42)
    n = 80
    return pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n) * 2,
        'f3': np.random.randn(n) + 10,
        'label': np.random.choice([0, 1, 2], n),
    })


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        'num_a': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        'num_b': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0],
        'cat_a': ['x', 'y', None, 'x', 'y', 'z', 'x', None, 'y', 'z'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def df_with_outliers():
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'target': np.random.choice([0, 1], n),
    })
    data.loc[0, 'f1'] = 100.0
    data.loc[1, 'f1'] = -100.0
    data.loc[2, 'f2'] = 50.0
    return data


@pytest.fixture
def imbalanced_df():
    np.random.seed(42)
    majority = pd.DataFrame({
        'f1': np.random.randn(90),
        'f2': np.random.randn(90),
        'target': [0] * 90,
    })
    minority = pd.DataFrame({
        'f1': np.random.randn(10) + 3,
        'f2': np.random.randn(10) + 3,
        'target': [1] * 10,
    })
    return pd.concat([majority, minority], ignore_index=True)


@pytest.fixture
def correlated_df():
    np.random.seed(42)
    n = 100
    base = np.random.randn(n)
    return pd.DataFrame({
        'f1': base,
        'f2': base + np.random.randn(n) * 0.01,
        'f3': np.random.randn(n) * 10,
        'target': np.random.choice([0, 1], n),
    })


@pytest.fixture
def low_variance_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'constant': [5.0] * n,
        'near_constant': [1.0] * 99 + [1.001],
        'varied': np.random.randn(n),
        'target': np.random.choice([0, 1], n),
    })
