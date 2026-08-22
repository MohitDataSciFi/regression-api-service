import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError, validator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from unittest.mock import Mock, patch

# Import the module under test
import sys
sys.path.insert(0, 'src')
from model_manager import (
    TrainingRequest,
    TrainingResponse,
    ErrorResponse,
    ModelManager,
    logger
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(10, 2, n_samples) + 2 * np.random.normal(0, 1, n_samples)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_bytes(sample_dataframe):
    """Convert sample dataframe to CSV bytes."""
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode()


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with temp directory."""
    return ModelManager(model_dir=str(tmp_path))


@pytest.fixture
def valid_training_request():
    """Create a valid training request."""
    return TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=['feature1', 'feature2', 'feature3']
    )


# =============================================================================
# Test ModelManager initialization and validation
# =============================================================================

def test_model_manager_initialization(tmp_path):
    """Test ModelManager initialization creates model directory."""
    model_dir = tmp_path / "models"
    manager = ModelManager(model_dir=str(model_dir))
    
    assert Path(model_dir).exists()
    assert manager.model_dir == str(model_dir)
    assert manager.model is None
    assert manager.metrics is None


def test_training_request_validation():
    """Test TrainingRequest validation rules."""
    # Valid request
    valid_request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    assert valid_request.target_column == 'target'
    assert valid_request.test_size == 0.2
    assert valid_request.random_state == 42
    assert valid_request.feature_columns is None

    # Invalid: empty target column
    with pytest.raises(ValidationError):
        TrainingRequest(target_column='', test_size=0.2, random_state=42)

    # Invalid: test_size out of range
    with pytest.raises(ValidationError):
        TrainingRequest(target_column='target', test_size=0.6, random_state=42)

    # Invalid: negative random_state
    with pytest.raises(ValidationError):
        TrainingRequest(target_column='target', test_size=0.2, random_state=-1)

    # Invalid: empty feature_columns
    with pytest.raises(ValidationError):
        TrainingRequest(
            target_column='target',
            test_size=0.2,
            random_state=42,
            feature_columns=[]
        )

    # Invalid: duplicate feature_columns
    with pytest.raises(ValidationError):
        TrainingRequest(
            target_column='target',
            test_size=0.2,
            random_state=42,
            feature_columns=['feature1', 'feature1']
        )


# =============================================================================
# Test ModelManager training logic
# =============================================================================

def test_train_model_success(model_manager, sample_dataframe, valid_training_request):
    """Test successful model training with valid data."""
    # Train the model
    result = model_manager.train(
        df=sample_dataframe,
        request=valid_training_request
    )

    # Verify result structure
    assert isinstance(result, TrainingResponse)
    assert result.model_id is not None
    assert result.training_timestamp is not None
    assert result.r_squared > 0
    assert result.adjusted_r_squared > 0
    assert result.mse > 0
    assert result.rmse > 0
    assert result.mae > 0
    assert len(result.coefficients) == 3  # 3 features
    assert len(result.p_values) == 3
    assert len(result.feature_importance) == 3
    assert 'breusch_pagan' in result.diagnostics
    assert 'durbin_watson' in result.diagnostics
    assert result.data_shape == {'rows': 100, 'features': 3}
    
    # Verify model was saved
    model_path = Path(result.model_path)
    assert model_path.exists()
    assert model_path.suffix == '.joblib'
    
    # Verify model can be loaded
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None


def test_train_model_without_feature_columns(model_manager, sample_dataframe):
    """Test training when feature_columns is None (uses all except target)."""
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=None
    )
    
    result = model_manager.train(df=sample_dataframe, request=request)
    
    # Should use all 3 features
    assert len(result.coefficients) == 3
    assert len(result.feature_importance) == 3


def test_train_model_invalid_target_column(model_manager, sample_dataframe):
    """Test training with non-existent target column."""
    request = TrainingRequest(
        target_column='nonexistent_column',
        test_size=0.2,
        random_state=42
    )
    
    with pytest.raises(ValueError, match="Target column 'nonexistent_column' not found"):
        model_manager.train(df=sample_dataframe, request=request)


def test_train_model_invalid_feature_columns(model_manager, sample_dataframe):
    """Test training with non-existent feature columns."""
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=['nonexistent_feature']
    )
    
    with pytest.raises(ValueError, match="Feature columns not found"):
        model_manager.train(df=sample_dataframe, request=request)


def test_train_model_insufficient_data(model_manager):
    """Test training with insufficient data."""
    # Create tiny dataframe
    small_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [1, 2, 3]
    })
    
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    
    with pytest.raises(ValueError, match="Insufficient data"):
        model_manager.train(df=small_df, request=request)


def test_train_model_constant_target(model_manager):
    """Test training with constant target variable."""
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 50),
        'target': np.ones(50)  # Constant target
    })
    
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    
    with pytest.raises(ValueError, match="Target variable has zero variance"):
        model_manager.train(df=df, request=request)


# =============================================================================
# Test ModelManager serialization
# =============================================================================

def test_model_serialization_roundtrip(model_manager, sample_dataframe, valid_training_request):
    """Test that model can be serialized and deserialized correctly."""
    # Train model
    result = model_manager.train(
        df=sample_dataframe,
        request=valid_training_request
    )
    
    # Load the saved model
    loaded_model = joblib.load(result.model_path)
    
    # Verify model makes predictions
    test_data = sample_dataframe[valid_training_request.feature_columns].iloc[:5]
    predictions = loaded_model.predict(test_data)
    
    assert len(predictions) == 5
    assert all(np.isfinite(predictions))


def test_model_manager_save_and_load(model_manager, sample_dataframe, valid_training_request):
    """Test explicit save and load methods."""
    # Train model
    result = model_manager.train(
        df=sample_dataframe,
        request=valid_training_request
    )
    
    # Save model explicitly
    save_path = model_manager.save_model()
    assert Path(save_path).exists()
    
    # Create new manager and load
    new_manager = ModelManager(model_dir=model_manager.model_dir)
    loaded_model = new_manager.load_model(result.model_id)
    
    assert loaded_model is not None
    assert hasattr(loaded_model, 'predict')


# =============================================================================
# Test error handling and edge cases
# =============================================================================

def test_train_model_with_missing_values(model_manager):
    """Test training with missing values in data."""
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [1, 2, 3, np.nan, 5],
        'target': [1, 2, 3, 4, 5]
    })
    
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    
    with pytest.raises(ValueError, match="Missing values"):
        model_manager.train(df=df, request=request)


def test_train_model_with_non_numeric_data(model_manager):
    """Test training with non-numeric data."""
    df = pd.DataFrame({
        'feature1': ['a', 'b', 'c', 'd', 'e'],
        'target': [1, 2, 3, 4, 5]
    })
    
    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    
    with pytest.raises(ValueError, match="non-numeric"):
        model_manager.train(df=df, request=request)


def test_model_manager_logging(model_manager, sample_dataframe, valid_training_request, caplog):
    """Test that training logs appropriate messages."""
    with caplog.at_level(logging.INFO):
        model_manager.train(
            df=sample_dataframe,
            request=valid_training_request
        )
    
    # Verify log messages
    log_messages = [record.message for record in caplog.records]
    assert any("Starting model training" in msg for msg in log_messages)
    assert any("Model training completed" in msg for msg in log_messages)


def test_model_manager_error_logging(model_manager, sample_dataframe, caplog):
    """Test that errors are logged properly."""
    request = TrainingRequest(
        target_column='nonexistent',
        test_size=0.2,
        random_state=42
    )
    
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            model_manager.train(df=sample_dataframe, request=request)
    
    log_messages = [record.message for record in caplog.records]
    assert any("Error during training" in msg for msg in log_messages)