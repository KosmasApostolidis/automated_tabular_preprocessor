def test_package_imports():
    from automated_tabular_preprocessor import (
        AbstractPreprocessor,
        TabularDataPreprocessor,
        AugmentedDataPreprocessor,
    )
    assert AbstractPreprocessor is not None
    assert TabularDataPreprocessor is not None
    assert AugmentedDataPreprocessor is not None


def test_version():
    import automated_tabular_preprocessor
    assert automated_tabular_preprocessor.__version__ == "0.1.0"
