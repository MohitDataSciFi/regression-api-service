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
from unittest.mock import patch, MagicMock

# Import the actual module (adjust path as needed)
import sys
sys.path.insert(0, 'src')
from model_manager import (
    ModelManager,
    ModelConfig,
    TrainingRequest,
    TrainingResponse,
    logger
)


@pytest.fixture
def sample_dataframe():
    """Create a sample regression dataset."""
    np.random.seed(42)
    n_samples = 100
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = 2 * X1 + 3 * X2 + np.random.randn(n_samples) * 0.1
    return pd.DataFrame({
        'feature1': X1,
        'feature2': X2,
        'target': y
    })


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with temp directory."""
    return ModelManager(model_dir=str(tmp_path))


@pytest.fixture
def valid_training_request():
    """Create a valid training request."""
    return TrainingRequest(
        target_column='target',
        feature_columns=['feature1', 'feature2'],
        model_config=ModelConfig(test_size=0.2, random_state=42)
    )


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = ModelConfig(test_size=0.2, random_state=42)
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.fit_intercept is True
        assert config.normalize is False

    def test_invalid_test_size(self):
        """Test invalid test_size values."""
        with pytest.raises(ValidationError):
            ModelConfig(test_size=0.05)  # Too small
        with pytest.raises(ValidationError):
            ModelConfig(test_size=0.6)   # Too large

    def test_invalid_random_state(self):
        """Test negative random_state."""
        with pytest.raises(ValidationError):
            ModelConfig(random_state=-1)


class TestTrainingRequest:
    """Tests for TrainingRequest validation."""

    def test_valid_request(self):
        """Test valid training request."""
        request = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )
        assert request.target_column == 'target'
        assert request.feature_columns == ['feature1', 'feature2']

    def test_empty_target_column(self):
        """Test empty target column."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column='', feature_columns=['feature1'])

    def test_whitespace_target_column(self):
        """Test whitespace-only target column."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column='   ', feature_columns=['feature1'])

    def test_empty_feature_columns(self):
        """Test empty feature columns list."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column='target',
                feature_columns=[]
            )

    def test_duplicate_feature_columns(self):
        """Test duplicate feature columns."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column='target',
                feature_columns=['feature1', 'feature1']
            )

    def test_whitespace_feature_column(self):
        """Test whitespace-only feature column."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column='target',
                feature_columns=['feature1', '   ']
            )

    def test_default_feature_columns(self):
        """Test None feature columns defaults to None."""
        request = TrainingRequest(target_column='target')
        assert request.feature_columns is None


class TestModelManagerTraining:
    """Tests for ModelManager training functionality."""

    def test_train_model_success(self, model_manager, sample_dataframe, valid_training_request):
        """Test successful model training."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert 'model_id' in result
        assert 'metrics' in result
        assert 'coefficients' in result
        assert 'p_values' in result
        assert 'r_squared' in result
        assert 'adjusted_r_squared' in result
        assert 'mse' in result
        assert 'rmse' in result
        assert 'mae' in result
        assert 'training_samples' in result
        assert 'test_samples' in result
        assert 'timestamp' in result
        assert 'model_path' in result

        # Verify metrics values
        assert result['r_squared'] > 0.9  # High R² for clean data
        assert result['mse'] < 1.0
        assert result['rmse'] < 1.0
        assert result['mae'] < 1.0
        assert result['training_samples'] == 80  # 80% of 100
        assert result['test_samples'] == 20      # 20% of 100

        # Verify coefficients
        assert 'feature1' in result['coefficients']
        assert 'feature2' in result['coefficients']
        assert abs(result['coefficients']['feature1'] - 2.0) < 0.5
        assert abs(result['coefficients']['feature2'] - 3.0) < 0.5

        # Verify p-values exist
        assert 'feature1' in result['p_values']
        assert 'feature2' in result['p_values']
        assert result['p_values']['feature1'] < 0.05
        assert result['p_values']['feature2'] < 0.05

        # Verify model file exists
        assert Path(result['model_path']).exists()

    def test_train_model_with_default_features(self, model_manager, sample_dataframe):
        """Test training with all columns except target as features."""
        request = TrainingRequest(target_column='target')
        result = model_manager.train_model(df=sample_dataframe, request=request)

        # Should use all columns except target
        assert 'feature1' in result['coefficients']
        assert 'feature2' in result['coefficients']
        assert 'target' not in result['coefficients']

    def test_train_model_missing_target(self, model_manager, sample_dataframe):
        """Test training with non-existent target column."""
        request = TrainingRequest(
            target_column='nonexistent',
            feature_columns=['feature1']
        )
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            model_manager.train_model(df=sample_dataframe, request=request)

    def test_train_model_missing_feature(self, model_manager, sample_dataframe):
        """Test training with non-existent feature column."""
        request = TrainingRequest(
            target_column='target',
            feature_columns=['nonexistent']
        )
        with pytest.raises(ValueError, match="Feature column 'nonexistent' not found"):
            model_manager.train_model(df=sample_dataframe, request=request)

    def test_train_model_serialization(self, model_manager, sample_dataframe, valid_training_request):
        """Test that model is properly serialized and can be loaded."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # Load the serialized model
        loaded_model = joblib.load(result['model_path'])
        assert loaded_model is not None

        # Verify model can make predictions
        test_data = sample_dataframe[['feature1', 'feature2']].iloc[:5]
        predictions = loaded_model.predict(test_data)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    def test_train_model_reproducibility(self, model_manager, sample_dataframe, valid_training_request):
        """Test that training is reproducible with same random state."""
        result1 = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )
        result2 = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # Same model_id should be generated for same data and config
        assert result1['model_id'] == result2['model_id']
        assert result1['coefficients'] == result2['coefficients']
        assert result1['metrics'] == result2['metrics']

    def test_train_model_different_random_state(self, model_manager, sample_dataframe):
        """Test that different random states produce different splits."""
        request1 = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2'],
            model_config=ModelConfig(random_state=42)
        )
        request2 = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2'],
            model_config=ModelConfig(random_state=43)
        )

        result1 = model_manager.train_model(df=sample_dataframe, request=request1)
        result2 = model_manager.train_model(df=sample_dataframe, request=request2)

        # Different random states should produce different model_ids
        assert result1['model_id'] != result2['model_id']

    def test_train_model_with_constant_feature(self, model_manager):
        """Test training with a constant feature column."""
        np.random.seed(42)
        df = pd.DataFrame({
            'constant': np.ones(100),
            'feature': np.random.randn(100),
            'target': 2 * np.random.randn(100) + 1
        })

        request = TrainingRequest(
            target_column='target',
            feature_columns=['constant', 'feature']
        )

        # Should handle constant features gracefully
        result = model_manager.train_model(df=df, request=request)
        assert 'constant' in result['coefficients']
        assert 'feature' in result['coefficients']

    def test_train_model_small_dataset(self, model_manager):
        """Test training with very small dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'target': np.random.randn(10)
        })

        request = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2'],
            model_config=ModelConfig(test_size=0.2)
        )

        # Should handle small datasets
        result = model_manager.train_model(df=df, request=request)
        assert result['training_samples'] == 8
        assert result['test_samples'] == 2

    def test_train_model_logging(self, model_manager, sample_dataframe, valid_training_request, caplog):
        """Test that training logs appropriate messages."""
        with caplog.at_level(logging.INFO):
            model_manager.train_model(
                df=sample_dataframe,
                request=valid_training_request
            )

        # Verify log messages
        assert any("Training model" in record.message for record in caplog.records)
        assert any("Model trained successfully" in record.message for record in caplog.records)
        assert any("R²" in record.message for record in caplog.records)


class TestModelManagerErrorHandling:
    """Tests for error handling in ModelManager."""

    def test_train_model_with_nan_values(self, model_manager):
        """Test training with NaN values in data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        df.loc[0, 'feature1'] = np.nan

        request = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        with pytest.raises(ValueError, match="Data contains NaN values"):
            model_manager.train_model(df=df, request=request)

    def test_train_model_with_inf_values(self, model_manager):
        """Test training with infinite values in data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        df.loc[0, 'feature1'] = np.inf

        request = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        with pytest.raises(ValueError, match="Data contains infinite values"):
            model_manager.train_model(df=df, request=request)

    def test_train_model_with_string_values(self, model_manager):
        """Test training with string values in numeric columns."""
        df = pd.DataFrame({
            'feature1': ['a', 'b', 'c'],
            'feature2': [1, 2, 3],
            'target': [1, 2, 3]
        })

        request = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        with pytest.raises(ValueError, match="Data contains non-numeric values"):
            model_manager.train_model(df=df, request=request)

    def test_train_model_empty_dataframe(self, model_manager):
        """Test training with empty dataframe."""
        df = pd.DataFrame()

        request = TrainingRequest(
            target_column='target',
            feature_columns=['feature1']
        )

        with pytest.raises(ValueError, match="DataFrame is empty"):
            model_manager.train_model(df=df, request=request)


class TestModelManagerModelID:
    """Tests for model ID generation."""

    def test_model_id_uniqueness(self, model_manager, sample_dataframe):
        """Test that different datasets produce different model IDs."""
        request = TrainingRequest(
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        # Create slightly different dataset
        df2 = sample_dataframe.copy()
        df2['feature1'] = df2['feature1'] + 0.1

        result1 = model_manager.train_model(df=sample_dataframe, request=request)
        result2 = model_manager.train_model(df=df2, request=request)

        assert result1['model_id'] != result2['model_id']

    def test_model_id_format(self, model_manager, sample_dataframe, valid_training_request):
        """Test model ID format."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # Model ID should be a string containing timestamp
        model_id = result['model_id']
        assert isinstance(model_id, str)
        assert len(model_id) > 0
        # Should contain timestamp components
        assert any(char.isdigit() for char in model_id)


class TestModelManagerFileOperations:
    """Tests for file operations in ModelManager."""

    def test_model_file_created(self, model_manager, sample_dataframe, valid_training_request):
        """Test that model file is created in the specified directory."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        model_path = Path(result['model_path'])
        assert model_path.exists()
        assert model_path.parent == Path(model_manager.model_dir)

    def test_model_file_extension(self, model_manager, sample_dataframe, valid_training_request):
        """Test that model file has correct extension."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        assert result['model_path'].endswith('.joblib')

    def test_multiple_models_saved(self, model_manager, sample_dataframe, valid_training_request):
        """Test that multiple models can be saved without overwriting."""
        result1 = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )
        result2 = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        assert result1['model_path'] != result2['model_path']
        assert Path(result1['model_path']).exists()
        assert Path(result2['model_path']).exists()


class TestModelManagerMetrics:
    """Tests for model metrics computation."""

    def test_metrics_quality(self, model_manager, sample_dataframe, valid_training_request):
        """Test that metrics are of high quality for clean data."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # For clean linear data, R² should be very high
        assert result['r_squared'] > 0.95
        assert result['adjusted_r_squared'] > 0.95

        # MSE should be small
        assert result['mse'] < 0.1

        # RMSE should be small
        assert result['rmse'] < 0.3

        # MAE should be small
        assert result['mae'] < 0.3

    def test_metrics_relationship(self, model_manager, sample_dataframe, valid_training_request):
        """Test relationships between metrics."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # RMSE should be >= MAE
        assert result['rmse'] >= result['mae']

        # RMSE should be sqrt of MSE
        assert abs(result['rmse'] - np.sqrt(result['mse'])) < 1e-10

        # Adjusted R² should be <= R²
        assert result['adjusted_r_squared'] <= result['r_squared']

    def test_p_values_significance(self, model_manager, sample_dataframe, valid_training_request):
        """Test that p-values are significant for true features."""
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # Both features should have significant p-values (< 0.05)
        assert result['p_values']['feature1'] < 0.05
        assert result['p_values']['feature2'] < 0.05


class TestModelManagerIntegration:
    """Integration tests for ModelManager."""

    def test_full_training_pipeline(self, model_manager, sample_dataframe, valid_training_request):
        """Test complete training pipeline end-to-end."""
        # Train model
        result = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request
        )

        # Load model and make predictions
        model = joblib.load(result['model_path'])
        test_data = sample_dataframe[['feature1', 'feature2']].iloc[:10]
        predictions = model.predict(test_data)

        # Verify predictions are reasonable
        actual = sample_dataframe['target'].iloc[:10]
        assert len(predictions) == 10
        assert np.allclose(predictions, actual, atol=1.0)

    def test_model_manager_with_csv_data(self, model_manager, sample_dataframe, valid_training_request):
        """Test training with CSV data."""
        # Convert to CSV
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Parse CSV back to DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Train model
        result = model_manager.train_model(
            df=df,
            request=valid_training_request
        )

        assert result['r_squared'] > 0.9