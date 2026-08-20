import io
import joblib
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.model_manager import ModelManager, TrainingRequest, ModelMetrics, TrainingResponse


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = 2 * X1 + 3 * X2 + np.random.randn(n_samples) * 0.1
    return pd.DataFrame({"feature1": X1, "feature2": X2, "target": y})


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with a temporary directory."""
    return ModelManager(model_dir=str(tmp_path / "models"))


def test_training_request_validation():
    """Test TrainingRequest validation logic."""
    # Valid request
    valid_request = TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"]
    )
    assert valid_request.target_column == "target"
    assert valid_request.test_size == 0.2

    # Invalid: empty target column
    with pytest.raises(ValueError):
        TrainingRequest(target_column="   ", test_size=0.2)

    # Invalid: test_size out of range
    with pytest.raises(ValueError):
        TrainingRequest(target_column="target", test_size=0.6)

    # Invalid: duplicate feature columns
    with pytest.raises(ValueError):
        TrainingRequest(
            target_column="target",
            feature_columns=["feature1", "feature1"]
        )

    # Invalid: empty feature columns list
    with pytest.raises(ValueError):
        TrainingRequest(
            target_column="target",
            feature_columns=[]
        )


def test_model_manager_training_and_metrics(model_manager, sample_data):
    """Test that ModelManager trains a model and returns valid metrics."""
    # Train the model
    result = model_manager.train(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42
    )

    # Verify the result structure
    assert result["success"] is True
    metrics = result["metrics"]
    assert isinstance(metrics, ModelMetrics)

    # Verify metrics values
    assert 0.0 <= metrics.r2_train <= 1.0
    assert 0.0 <= metrics.r2_test <= 1.0
    assert metrics.mse_train >= 0
    assert metrics.mse_test >= 0
    assert metrics.mae_train >= 0
    assert metrics.mae_test >= 0

    # Verify coefficients and p-values
    assert "feature1" in metrics.coefficients
    assert "feature2" in metrics.coefficients
    assert "feature1" in metrics.p_values
    assert "feature2" in metrics.p_values

    # Verify sample counts
    assert metrics.training_samples == 80  # 80% of 100
    assert metrics.test_samples == 20  # 20% of 100
    assert metrics.feature_count == 2

    # Verify model file was created
    assert Path(metrics.model_path).exists()


def test_model_manager_serialization(model_manager, sample_data):
    """Test that the trained model can be serialized and loaded."""
    # Train the model
    result = model_manager.train(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42
    )

    # Load the serialized model
    model_path = result["metrics"].model_path
    loaded_model = joblib.load(model_path)

    # Verify the loaded model works
    test_data = pd.DataFrame({
        "feature1": [0.5, -0.3],
        "feature2": [1.2, 0.7]
    })
    predictions = loaded_model.predict(test_data)
    assert len(predictions) == 2
    assert all(np.isfinite(predictions))


def test_model_manager_error_handling(model_manager, sample_data):
    """Test error handling for invalid training inputs."""
    # Test with non-existent target column
    with pytest.raises(KeyError):
        model_manager.train(
            data=sample_data,
            target_column="nonexistent_column",
            test_size=0.2,
            random_state=42
        )

    # Test with insufficient data
    small_data = sample_data.head(5)
    with pytest.raises(ValueError):
        model_manager.train(
            data=small_data,
            target_column="target",
            test_size=0.2,
            random_state=42
        )


def test_model_manager_diagnostics(model_manager, sample_data):
    """Test that diagnostic statistics are computed correctly."""
    # Train the model
    result = model_manager.train(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42
    )

    metrics = result["metrics"]

    # Verify diagnostic statistics are present and valid
    assert metrics.f_statistic > 0
    assert 0.0 <= metrics.f_p_value <= 1.0
    assert 0.0 <= metrics.durbin_watson <= 4.0
    assert 0.0 <= metrics.breusch_pagan_p_value <= 1.0

    # Verify p-values are reasonable (should be very small for our synthetic data)
    assert metrics.p_values["feature1"] < 0.05
    assert metrics.p_values["feature2"] < 0.05

    # Verify coefficients are close to true values (2 and 3)
    assert abs(metrics.coefficients["feature1"] - 2.0) < 0.5
    assert abs(metrics.coefficients["feature2"] - 3.0) < 0.5