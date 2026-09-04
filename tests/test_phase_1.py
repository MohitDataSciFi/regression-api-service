import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile

from src.model_manager import ModelManager, TrainingResponse


@pytest.fixture
def sample_data():
    """Create sample regression data."""
    np.random.seed(42)
    n_samples = 100
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = 2.5 * X1 + 1.5 * X2 + np.random.randn(n_samples) * 0.1
    return pd.DataFrame({"feature1": X1, "feature2": X2, "target": y})


@pytest.fixture
def model_manager(tmp_path):
    """Create ModelManager with temporary directory."""
    return ModelManager(model_dir=Path(tmp_path) / "models")


@pytest.mark.asyncio
async def test_train_model_returns_valid_metrics_and_serializes(model_manager, sample_data):
    """Test that training produces valid metrics and serializes model."""
    # Train model
    metrics, model_id = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        feature_columns=["feature1", "feature2"]
    )
    
    # Verify metrics
    assert "r2_score" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mse" in metrics
    assert metrics["r2_score"] > 0.9  # High R² for clean data
    
    # Verify model file exists
    model_path = model_manager.model_dir / f"{model_id}.joblib"
    assert model_path.exists()
    
    # Verify model can be loaded and makes predictions
    loaded_model = joblib.load(model_path)
    test_data = sample_data[["feature1", "feature2"]].iloc[:5]
    predictions = loaded_model.predict(test_data)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


@pytest.mark.asyncio
async def test_train_model_with_auto_feature_selection(model_manager, sample_data):
    """Test training when feature_columns is None (auto-select all except target)."""
    metrics, model_id = await model_manager.train_model(
        data=sample_data,
        target_column="target"
    )
    
    # Verify all non-target columns were used
    assert "feature1" in metrics.get("coefficients", {})
    assert "feature2" in metrics.get("coefficients", {})
    assert metrics["r2_score"] > 0.9


@pytest.mark.asyncio
async def test_train_model_handles_invalid_target_column(model_manager, sample_data):
    """Test that invalid target column raises appropriate error."""
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
        await model_manager.train_model(
            data=sample_data,
            target_column="nonexistent"
        )


@pytest.mark.asyncio
async def test_train_model_with_custom_test_size_and_seed(model_manager, sample_data):
    """Test training with custom test_size and random_state parameters."""
    metrics_1, model_id_1 = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.3,
        random_state=123
    )
    
    metrics_2, model_id_2 = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.3,
        random_state=123
    )
    
    # Same seed should produce same results
    assert metrics_1["r2_score"] == metrics_2["r2_score"]
    assert model_id_1 != model_id_2  # Different model IDs despite same data


@pytest.mark.asyncio
async def test_train_model_diagnostics_include_pvalues_and_coefficients(model_manager, sample_data):
    """Test that diagnostics include p-values and coefficient information."""
    metrics, model_id = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        feature_columns=["feature1", "feature2"]
    )
    
    # Check coefficients
    assert "coefficients" in metrics
    coeffs = metrics["coefficients"]
    assert "feature1" in coeffs
    assert "feature2" in coeffs
    assert "intercept" in coeffs
    
    # Check p-values
    assert "p_values" in metrics
    p_values = metrics["p_values"]
    assert "feature1" in p_values
    assert "feature2" in p_values
    assert "intercept" in p_values
    
    # Features should be significant (p < 0.05)
    assert p_values["feature1"] < 0.05
    assert p_values["feature2"] < 0.05
    
    # Check diagnostics dict
    assert "diagnostics" in metrics
    diagnostics = metrics["diagnostics"]
    assert "durbin_watson" in diagnostics
    assert "breusch_pagan_pvalue" in diagnostics
    assert "n_observations" in diagnostics
    assert "n_features" in diagnostics