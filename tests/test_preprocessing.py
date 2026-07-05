from sklearn.compose import ColumnTransformer

from src.automate import build_preprocessor


def test_build_preprocessor():
    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)
