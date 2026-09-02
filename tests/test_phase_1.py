import asyncio
import io
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.model_manager import ModelManager, TrainingRequest, TrainingResponse


@pytest.fixture
def sample_data():
    """Create sample regression data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.1
    df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    df["target"] = y
    return df


@pytest.fixture
def model_manager(tmp_path):
    """Create ModelManager instance with temp directory."""
    return ModelManager(model_dir=str(tmp_path / "models"))


@pytest.mark.asyncio
async def test_train_model_returns_valid_response(model_manager, sample_data):
    """Test that training produces valid response with correct metrics."""
    response = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42
    )
    
    assert isinstance(response, TrainingResponse)
    assert response.model_id
    assert response.r_squared > 0.9  # High R² for clean data
    assert response.adjusted_r_squared > 0.9
    assert response.mse < 0.1
    assert response.rmse < 0.3
    assert response.mae < 0.3
    assert response.n_samples == 100
    assert response.n_features == 3
    assert len(response.coefficients) == 4  # 3 features + intercept
    assert len(response.p_values) == 4
    assert Path(response.model_path).exists()


@pytest.mark.asyncio
async def test_train_model_serializes_and_loads_model(model_manager, sample_data):
    """Test that trained model is properly serialized and can be loaded."""
    response = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42
    )
    
    # Verify model file exists and can be loaded
    model_path = Path(response.model_path)
    assert model_path.exists()
    
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict")
    
    # Test prediction with loaded model
    test_features = sample_data[["feature1", "feature2", "feature3"]].iloc[:5]
    predictions = loaded_model.predict(test_features)
    assert len(predictions) == 5
    assert np.all(np.isfinite(predictions))


@pytest.mark.asyncio
async def test_train_model_handles_invalid_target_column(model_manager, sample_data):
    """Test that training raises error for invalid target column."""
    with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
        await model_manager.train_model(
            data=sample_data,
            target_column="nonexistent",
            test_size=0.2,
            random_state=42
        )


@pytest.mark.asyncio
async def test_train_model_handles_non_numeric_data(model_manager):
    """Test that training handles data with non-numeric columns."""
    data = pd.DataFrame({
        "numeric1": [1.0, 2.0, 3.0, 4.0],
        "numeric2": [2.0, 4.0, 6.0, 8.0],
        "categorical": ["A", "B", "A", "B"],
        "target": [3.0, 6.0, 9.0, 12.0]
    })
    
    with pytest.raises(ValueError, match="non-numeric"):
        await model_manager.train_model(
            data=data,
            target_column="target",
            test_size=0.2,
            random_state=42
        )


@pytest.mark.asyncio
async def test_train_model_consistency_with_same_seed(model_manager, sample_data):
    """Test that training with same seed produces consistent results."""
    response1 = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42
    )
    
    response2 = await model_manager.train_model(
        data=sample_data,
        target_column="target",
        test_size=0.2,
        random_state=42
    )
    
    assert response1.r_squared == response2.r_squared
    assert response1.coefficients == response2.coefficients
    assert response1.p_values == response2.p_values
    assert response1.mse == response2.mse