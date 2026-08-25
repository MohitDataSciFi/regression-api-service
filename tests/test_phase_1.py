import asyncio
import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.model_manager import (
    MODEL_DIR,
    ModelManager,
    TrainingRequest,
    TrainingResponse,
)


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
    return ModelManager(model_dir=tmp_path)


@pytest.mark.asyncio
async def test_train_model_success(model_manager, sample_data):
    """Test successful model training with valid data."""
    # Train model
    response = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"],
    )

    # Verify response type and structure
    assert isinstance(response, TrainingResponse)
    assert response.model_id is not None
    assert response.model_path is not None
    assert Path(response.model_path).exists()
    assert "r2" in response.metrics
    assert "mae" in response.metrics
    assert "mse" in response.metrics
    assert "rmse" in response.metrics
    assert "coefficients" in response.diagnostics
    assert "p_values" in response.diagnostics
    assert "durbin_watson" in response.diagnostics
    assert "breusch_pagan" in response.diagnostics
    assert response.data_shape == {"rows": 100, "columns": 3}
    assert response.feature_importance == {"feature1": pytest.approx(2.0, abs=0.1), "feature2": pytest.approx(3.0, abs=0.1)}


@pytest.mark.asyncio
async def test_train_model_without_feature_columns(model_manager, sample_data):
    """Test training when feature_columns is None (uses all other columns)."""
    response = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=None,
    )

    assert isinstance(response, TrainingResponse)
    assert response.data_shape == {"rows": 100, "columns": 3}
    assert set(response.feature_importance.keys()) == {"feature1", "feature2"}


@pytest.mark.asyncio
async def test_train_model_invalid_target_column(model_manager, sample_data):
    """Test training with a non-existent target column."""
    with pytest.raises(HTTPException) as exc_info:
        await model_manager.train_model(
            data=sample_data,
            target_column="nonexistent_column",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1"],
        )
    assert exc_info.value.status_code == 400
    assert "Target column 'nonexistent_column' not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_train_model_serialization(model_manager, sample_data):
    """Test that the trained model is properly serialized and can be loaded."""
    response = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"],
    )

    # Load the serialized model
    loaded_model = joblib.load(response.model_path)
    assert hasattr(loaded_model, "predict")
    assert hasattr(loaded_model, "coef_")
    assert hasattr(loaded_model, "intercept_")

    # Verify model can make predictions
    test_data = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]})
    predictions = loaded_model.predict(test_data)
    assert len(predictions) == 2
    assert all(np.isfinite(predictions))


@pytest.mark.asyncio
async def test_training_request_validation():
    """Test Pydantic validation for TrainingRequest."""
    # Valid request
    valid_request = TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"],
    )
    assert valid_request.target_column == "target"

    # Invalid: empty target column
    with pytest.raises(ValidationError):
        TrainingRequest(target_column="", test_size=0.2, random_state=42)

    # Invalid: test_size out of range
    with pytest.raises(ValidationError):
        TrainingRequest(target_column="target", test_size=0.6, random_state=42)

    # Invalid: duplicate feature columns
    with pytest.raises(ValidationError):
        TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "feature1"],
        )

    # Invalid: empty feature columns list
    with pytest.raises(ValidationError):
        TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=[],
        )


@pytest.mark.asyncio
async def test_train_model_with_missing_features(model_manager, sample_data):
    """Test training when specified feature columns don't exist in data."""
    with pytest.raises(HTTPException) as exc_info:
        await model_manager.train_model(
            data=sample_data,
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "nonexistent_feature"],
        )
    assert exc_info.value.status_code == 400
    assert "Feature columns not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_train_model_concurrent_access(model_manager, sample_data):
    """Test that concurrent training requests are handled safely."""
    async def train():
        return await model_manager.train_model(
            data=sample_data,
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "feature2"],
        )

    # Run multiple training requests concurrently
    results = await asyncio.gather(*[train() for _ in range(3)])
    
    # All should succeed and produce valid responses
    for response in results:
        assert isinstance(response, TrainingResponse)
        assert Path(response.model_path).exists()