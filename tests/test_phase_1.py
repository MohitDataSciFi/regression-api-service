import io
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.model_manager import ModelManager, TrainingRequest, TrainingResponse


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(0, 1, n_samples) + 2 * np.random.normal(0, 1, n_samples)
    })
    return data


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with a temporary directory."""
    return ModelManager(model_dir=Path(tmp_path))


@pytest.fixture
def valid_training_request():
    """Create a valid training request."""
    return TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=['feature1', 'feature2']
    )


@pytest.mark.asyncio
async def test_train_model_returns_valid_response(model_manager, sample_data, valid_training_request):
    """Test that training returns a valid TrainingResponse with expected metrics."""
    # Act
    response = await model_manager.train_model(
        data=sample_data,
        target_column=valid_training_request.target_column,
        test_size=valid_training_request.test_size,
        random_state=valid_training_request.random_state,
        feature_columns=valid_training_request.feature_columns
    )

    # Assert
    assert isinstance(response, TrainingResponse)
    assert response.model_id is not None
    assert response.timestamp is not None
    assert response.r_squared > 0.5  # Should have decent R² with correlated data
    assert response.n_samples == 100
    assert response.n_features == 2
    assert set(response.coefficients.keys()) == {'feature1', 'feature2'}
    assert set(response.p_values.keys()) == {'feature1', 'feature2'}
    assert response.mse > 0
    assert response.rmse > 0
    assert response.mae > 0
    assert response.f_statistic > 0
    assert response.f_p_value < 0.05  # Model should be significant
    assert response.model_path.endswith('.joblib')


@pytest.mark.asyncio
async def test_train_model_serializes_model_file(model_manager, sample_data, valid_training_request):
    """Test that the trained model is actually saved to disk."""
    # Act
    response = await model_manager.train_model(
        data=sample_data,
        target_column=valid_training_request.target_column,
        test_size=valid_training_request.test_size,
        random_state=valid_training_request.random_state,
        feature_columns=valid_training_request.feature_columns
    )

    # Assert
    model_path = Path(response.model_path)
    assert model_path.exists()
    assert model_path.is_file()
    
    # Verify the model can be loaded and makes predictions
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, 'predict')
    
    # Test prediction
    test_data = sample_data[valid_training_request.feature_columns].iloc[:5]
    predictions = loaded_model.predict(test_data)
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


@pytest.mark.asyncio
async def test_train_model_with_auto_feature_selection(model_manager, sample_data):
    """Test that training works when feature_columns is None (uses all other columns)."""
    # Arrange
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=None
    )

    # Act
    response = await model_manager.train_model(
        data=sample_data,
        target_column=request.target_column,
        test_size=request.test_size,
        random_state=request.random_state,
        feature_columns=request.feature_columns
    )

    # Assert
    assert response.n_features == 2  # Should use both feature columns
    assert set(response.coefficients.keys()) == {'feature1', 'feature2'}


@pytest.mark.asyncio
async def test_train_model_handles_missing_target_column(model_manager, sample_data):
    """Test that training raises appropriate error for missing target column."""
    # Arrange
    request = TrainingRequest(
        target_column='nonexistent_column',
        test_size=0.2,
        random_state=42
    )

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await model_manager.train_model(
            data=sample_data,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            feature_columns=request.feature_columns
        )
    
    assert exc_info.value.status_code == 400
    assert "Target column" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_train_model_handles_invalid_feature_columns(model_manager, sample_data):
    """Test that training raises appropriate error for invalid feature columns."""
    # Arrange
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=['nonexistent_feature']
    )

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await model_manager.train_model(
            data=sample_data,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            feature_columns=request.feature_columns
        )
    
    assert exc_info.value.status_code == 400
    assert "Feature column" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_train_model_handles_insufficient_data(model_manager):
    """Test that training raises appropriate error for insufficient data."""
    # Arrange
    small_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [7, 8, 9]
    })
    request = TrainingRequest(
        target_column='target',
        test_size=0.5,
        random_state=42
    )

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await model_manager.train_model(
            data=small_data,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            feature_columns=request.feature_columns
        )
    
    assert exc_info.value.status_code == 400
    assert "Insufficient data" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_train_model_handles_nan_values(model_manager, sample_data):
    """Test that training handles NaN values in the data."""
    # Arrange
    data_with_nan = sample_data.copy()
    data_with_nan.loc[0, 'feature1'] = np.nan
    data_with_nan.loc[1, 'target'] = np.nan
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )

    # Act
    response = await model_manager.train_model(
        data=data_with_nan,
        target_column=request.target_column,
        test_size=request.test_size,
        random_state=request.random_state,
        feature_columns=request.feature_columns
    )

    # Assert
    assert response.n_samples < 100  # Should have dropped rows with NaN
    assert response.r_squared > 0.5


@pytest.mark.asyncio
async def test_training_request_validation():
    """Test Pydantic validation for TrainingRequest."""
    # Test valid request
    valid_request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    assert valid_request.target_column == 'target'
    assert valid_request.test_size == 0.2
    assert valid_request.random_state == 42

    # Test invalid test_size
    with pytest.raises(Exception):
        TrainingRequest(
            target_column='target',
            test_size=0.6,  # Too large
            random_state=42
        )

    # Test invalid target_column
    with pytest.raises(Exception):
        TrainingRequest(
            target_column='',  # Empty
            test_size=0.2,
            random_state=42
        )

    # Test duplicate feature columns
    with pytest.raises(Exception):
        TrainingRequest(
            target_column='target',
            test_size=0.2,
            random_state=42,
            feature_columns=['feature1', 'feature1']  # Duplicate
        )