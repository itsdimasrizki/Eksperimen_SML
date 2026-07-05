from src.mlflow_tracking import extract_baseline_parameters


def test_extract_baseline_parameters():
    metadata = {
        "model_name": "RandomForest",
        "version": "baseline",
        "random_state": 42,
    }

    params = extract_baseline_parameters(metadata)

    assert params["version"] == "baseline"
    assert params["random_state"] == 42