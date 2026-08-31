import asyncio
import io
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.model_manager import (
    MODEL_DIR,
    ModelManager,
    TrainingRequest,
    TrainingResponse,
    logger,
)


@pytest.fixture
def sample_csv_data() -> bytes:
    """Create a sample CSV dataset for testing."""
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randn(100) * 2 + 1,
        }
    )
    return df.to_csv(index=False).encode()


@pytest.fixture
def model_manager(tmp_path: Path) -> ModelManager:
    """Create a ModelManager instance with a temporary directory."""
    return ModelManager(model_dir=tmp_path)


@pytest.fixture
def valid_training_request() -> TrainingRequest:
    """Create a valid training request."""
    return TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        model_name="test_model",
    )


def test_training_request_validation():
    """Test Pydantic validation for TrainingRequest."""
    # Valid request
    request = TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        model_name="test_model",
    )
    assert request.target_column == "target"
    assert request.test_size == 0.2

    # Invalid target column (empty)
    with pytest.raises(ValueError, match="Target column cannot be empty"):
        TrainingRequest(target_column="   ", test_size=0.2, random_state=42, model_name="test")

    # Invalid model name (special characters)
    with pytest.raises(ValueError, match="Model name must be alphanumeric"):
        TrainingRequest(target_column="target", test_size=0.2, random_state=42, model_name="test-model!")

    # Invalid test_size (out of range)
    with pytest.raises(ValueError):
        TrainingRequest(target_column="target", test_size=0.05, random_state=42, model_name="test")


@pytest.mark.asyncio
async def test_train_model_success(
    model_manager: ModelManager,
    sample_csv_data: bytes,
    valid_training_request: TrainingRequest,
):
    """Test successful model training with valid data."""
    response = await model_manager.train_model(
        file_content=sample_csv_data,
        target_column=valid_training_request.target_column,
        test_size=valid_training_request.test_size,
        random_state=valid_training_request.random_state,
        model_name=valid_training_request.model_name,
    )

    # Verify response structure
    assert isinstance(response, TrainingResponse)
    assert response.model_id == "test_model"
    assert response.model_path.endswith(".joblib")
    assert response.training_timestamp
    assert response.training_duration > 0

    # Verify metrics
    assert "r2" in response.metrics
    assert "mae" in response.metrics
    assert "mse" in response.metrics
    assert "rmse" in response.metrics
    assert 0 <= response.metrics["r2"] <= 1

    # Verify diagnostics
    assert "coefficients" in response.diagnostics
    assert "p_values" in response.diagnostics
    assert "durbin_watson" in response.diagnostics
    assert "breusch_pagan" in response.diagnostics
    assert len(response.diagnostics["coefficients"]) > 0
    assert len(response.diagnostics["p_values"]) > 0

    # Verify model file was saved
    model_path = Path(response.model_path)
    assert model_path.exists()
    assert model_path.suffix == ".joblib"

    # Verify model can be loaded and makes predictions
    loaded_model = joblib.load(model_path)
    test_data = pd.DataFrame(
        {"feature1": [0.5], "feature2": [-0.3]}
    )
    prediction = loaded_model.predict(test_data)
    assert prediction.shape == (1,)


@pytest.mark.asyncio
async def test_train_model_invalid_target_column(
    model_manager: ModelManager,
    sample_csv_data: bytes,
):
    """Test training with non-existent target column."""
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
        await model_manager.train_model(
            file_content=sample_csv_data,
            target_column="nonexistent",
            test_size=0.2,
            random_state=42,
            model_name="test_model",
        )


@pytest.mark.asyncio
async def test_train_model_insufficient_data(
    model_manager: ModelManager,
):
    """Test training with insufficient data for train/test split."""
    # Create tiny dataset (only 5 rows)
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(5),
            "feature2": np.random.randn(5),
            "target": np.random.randn(5),
        }
    )
    tiny_csv = df.to_csv(index=False).encode()

    with pytest.raises(ValueError, match="Insufficient data"):
        await model_manager.train_model(
            file_content=tiny_csv,
            target_column="target",
            test_size=0.2,
            random_state=42,
            model_name="test_model",
        )


@pytest.mark.asyncio
async def test_train_model_serialization_and_reproducibility(
    model_manager: ModelManager,
    sample_csv_data: bytes,
):
    """Test that training is reproducible and model serialization works."""
    # Train two models with same parameters
    response1 = await model_manager.train_model(
        file_content=sample_csv_data,
        target_column="target",
        test_size=0.2,
        random_state=42,
        model_name="model_a",
    )
    response2 = await model_manager.train_model(
        file_content=sample_csv_data,
        target_column="target",
        test_size=0.2,
        random_state=42,
        model_name="model_b",
    )

    # Verify same metrics (reproducibility)
    assert response1.metrics["r2"] == response2.metrics["r2"]
    assert response1.metrics["mae"] == response2.metrics["mae"]
    assert response1.diagnostics["coefficients"] == response2.diagnostics["coefficients"]

    # Verify models are saved separately
    assert response1.model_path != response2.model_path
    assert Path(response1.model_path).exists()
    assert Path(response2.model_path).exists()

    # Verify models produce same predictions
    model1 = joblib.load(response1.model_path)
    model2 = joblib.load(response2.model_path)
    test_data = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
    pred1 = model1.predict(test_data)
    pred2 = model2.predict(test_data)
    np.testing.assert_array_almost_equal(pred1, pred2)


@pytest.mark.asyncio
async def test_train_model_concurrent_safety(
    model_manager: ModelManager,
    sample_csv_data: bytes,
):
    """Test that concurrent training requests are handled safely."""
    # Run multiple training requests concurrently
    tasks = [
        model_manager.train_model(
            file_content=sample_csv_data,
            target_column="target",
            test_size=0.2,
            random_state=42,
            model_name=f"model_{i}",
        )
        for i in range(5)
    ]
    responses = await asyncio.gather(*tasks)

    # Verify all models were created successfully
    assert len(responses) == 5
    for response in responses:
        assert isinstance(response, TrainingResponse)
        assert Path(response.model_path).exists()
        assert response.metrics["r2"] > 0