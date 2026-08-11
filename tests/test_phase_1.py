import asyncio
import io
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.model_manager import ModelConfig, ModelManager, TrainingResponse


@pytest.fixture
def sample_data():
    """Create sample regression data."""
    np.random.seed(42)
    n_samples = 100
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = 2 * X1 + 3 * X2 + np.random.randn(n_samples) * 0.1
    return pd.DataFrame({"feature1": X1, "feature2": X2, "target": y})


@pytest.fixture
def model_manager(tmp_path):
    """Create ModelManager with temporary directory."""
    return ModelManager(model_dir=str(tmp_path / "models"))


@pytest.fixture
def valid_config():
    """Create valid model config."""
    return ModelConfig(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"]
    )


@pytest.mark.asyncio
async def test_train_model_success(model_manager, sample_data, valid_config):
    """Test successful model training with valid data and config."""
    metrics, model_path = await model_manager.train_model(sample_data, valid_config)
    
    # Verify metrics
    assert "r_squared" in metrics
    assert "adjusted_r_squared" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mse" in metrics
    assert metrics["r_squared"] > 0.9  # High R² for clean data
    
    # Verify coefficients
    assert "feature1" in metrics["coefficients"]
    assert "feature2" in metrics["coefficients"]
    assert abs(metrics["coefficients"]["feature1"] - 2.0) < 0.5
    assert abs(metrics["coefficients"]["feature2"] - 3.0) < 0.5
    
    # Verify p-values
    assert "feature1" in metrics["p_values"]
    assert "feature2" in metrics["p_values"]
    assert metrics["p_values"]["feature1"] < 0.05
    assert metrics["p_values"]["feature2"] < 0.05
    
    # Verify model file exists
    assert Path(model_path).exists()
    
    # Verify model can be loaded
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict")


@pytest.mark.asyncio
async def test_train_model_missing_target(model_manager, sample_data):
    """Test training with missing target column."""
    config = ModelConfig(target_column="nonexistent", test_size=0.2, random_state=42)
    
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
        await model_manager.train_model(sample_data, config)


@pytest.mark.asyncio
async def test_train_model_empty_data(model_manager, valid_config):
    """Test training with empty dataframe."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="Data is empty"):
        await model_manager.train_model(empty_df, valid_config)


@pytest.mark.asyncio
async def test_train_model_feature_columns_subset(model_manager, sample_data):
    """Test training with subset of feature columns."""
    config = ModelConfig(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1"]  # Only use one feature
    )
    
    metrics, model_path = await model_manager.train_model(sample_data, config)
    
    # Verify only specified feature is used
    assert "feature1" in metrics["coefficients"]
    assert "feature2" not in metrics["coefficients"]
    assert metrics["r_squared"] > 0.5  # Still decent with one feature


@pytest.mark.asyncio
async def test_train_model_serialization_and_reproducibility(model_manager, sample_data, valid_config):
    """Test model serialization and reproducibility."""
    # Train twice with same config
    metrics1, path1 = await model_manager.train_model(sample_data, valid_config)
    metrics2, path2 = await model_manager.train_model(sample_data, valid_config)
    
    # Verify same metrics (deterministic with fixed random_state)
    assert metrics1["r_squared"] == metrics2["r_squared"]
    assert metrics1["coefficients"] == metrics2["coefficients"]
    
    # Verify different model paths (timestamp-based)
    assert path1 != path2
    
    # Load and verify model works
    loaded_model = joblib.load(path1)
    test_data = sample_data[valid_config.feature_columns].iloc[:5]
    predictions = loaded_model.predict(test_data)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))