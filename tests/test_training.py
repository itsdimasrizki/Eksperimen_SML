from sklearn.ensemble import RandomForestClassifier

from src.modelling import build_baseline_model


def test_build_baseline_model():
    model = build_baseline_model()

    assert isinstance(model, RandomForestClassifier)
