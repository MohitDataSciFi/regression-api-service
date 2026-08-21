import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError, validator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.model_manager import (
    RegressionModelManager,
    TrainingRequest,
    TrainingMetrics,
    ModelInfo,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 2 + 1,
    }
    return pd.DataFrame(data)


@pytest.fixture
def model_manager(tmp_path):
    """Create a RegressionModelManager instance with temp directory."""
    return RegressionModelManager(model_dir=str(tmp_path))


@pytest.fixture
def valid_training_request():
    """Create a valid training request."""
    return TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"],
    )


class TestTrainingRequestValidation:
    """Test Pydantic validation for TrainingRequest."""

    def test_valid_request(self, valid_training_request):
        """Test that a valid request passes validation."""
        assert valid_training_request.target_column == "target"
        assert valid_training_request.test_size == 0.2
        assert valid_training_request.random_state == 42
        assert valid_training_request.feature_columns == ["feature1", "feature2"]

    def test_empty_target_column_raises_error(self):
        """Test that empty target column raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="   ",
                test_size=0.2,
                random_state=42,
            )

    def test_duplicate_feature_columns_raises_error(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                test_size=0.2,
                random_state=42,
                feature_columns=["feature1", "feature1"],
            )

    def test_invalid_test_size_raises_error(self):
        """Test that test_size outside [0.1, 0.5] raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                test_size=0.6,
                random_state=42,
            )


class TestRegressionModelManager:
    """Test RegressionModelManager core functionality."""

    def test_train_model_returns_metrics(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that train_model returns valid TrainingMetrics."""
        metrics = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        assert isinstance(metrics, TrainingMetrics)
        assert 0.0 <= metrics.r2_score <= 1.0
        assert metrics.mse >= 0
        assert metrics.rmse >= 0
        assert metrics.mae >= 0
        assert len(metrics.coefficients) == 2
        assert "feature1" in metrics.coefficients
        assert "feature2" in metrics.coefficients
        assert metrics.training_samples > 0
        assert metrics.test_samples > 0
        assert metrics.feature_count == 2
        assert metrics.training_duration_seconds > 0
        assert metrics.model_version
        assert metrics.timestamp

    def test_train_model_with_default_features(
        self, model_manager, sample_dataframe
    ):
        """Test training with default features (all except target)."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        metrics = model_manager.train_model(
            df=sample_dataframe,
            request=request,
        )

        assert metrics.feature_count == 2
        assert set(metrics.coefficients.keys()) == {"feature1", "feature2"}

    def test_train_model_p_values_are_valid(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that p-values are valid probabilities."""
        metrics = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        for feature, p_value in metrics.p_values.items():
            assert 0.0 <= p_value <= 1.0
            assert isinstance(p_value, float)

    def test_serialize_and_load_model(
        self, model_manager, sample_dataframe, valid_training_request, tmp_path
    ):
        """Test model serialization and loading."""
        # Train model
        metrics = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Serialize model
        model_path = model_manager.save_model(
            model=model_manager.model,
            metrics=metrics,
            feature_names=valid_training_request.feature_columns,
            target_name=valid_training_request.target_column,
        )

        assert Path(model_path).exists()

        # Load model
        loaded_model = model_manager.load_model(model_path)
        assert loaded_model is not None

        # Test prediction with loaded model
        test_data = sample_dataframe[valid_training_request.feature_columns].iloc[:5]
        predictions = loaded_model.predict(test_data)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    def test_save_model_creates_model_info(
        self, model_manager, sample_dataframe, valid_training_request, tmp_path
    ):
        """Test that save_model creates proper ModelInfo metadata."""
        metrics = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        model_path = model_manager.save_model(
            model=model_manager.model,
            metrics=metrics,
            feature_names=valid_training_request.feature_columns,
            target_name=valid_training_request.target_column,
        )

        # Check that metadata file was created
        metadata_path = Path(model_path).with_suffix(".json")
        assert metadata_path.exists()

        # Load and verify metadata
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["model_type"] == "linear_regression"
        assert metadata["target_name"] == "target"
        assert metadata["feature_names"] == ["feature1", "feature2"]
        assert metadata["metrics"]["r2_score"] == metrics.r2_score
        assert metadata["model_id"]
        assert metadata["created_at"]

    def test_train_model_with_missing_target_column(
        self, model_manager, sample_dataframe
    ):
        """Test that missing target column raises error."""
        request = TrainingRequest(
            target_column="nonexistent_target",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(KeyError):
            model_manager.train_model(
                df=sample_dataframe,
                request=request,
            )

    def test_train_model_with_missing_feature_column(
        self, model_manager, sample_dataframe
    ):
        """Test that missing feature column raises error."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "nonexistent_feature"],
        )

        with pytest.raises(KeyError):
            model_manager.train_model(
                df=sample_dataframe,
                request=request,
            )

    def test_train_model_reproducibility(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training is reproducible with same random state."""
        metrics1 = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        metrics2 = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        assert metrics1.r2_score == metrics2.r2_score
        assert metrics1.coefficients == metrics2.coefficients
        assert metrics1.intercept == metrics2.intercept
        assert metrics1.training_samples == metrics2.training_samples
        assert metrics1.test_samples == metrics2.test_samples

    def test_train_model_with_different_random_state(
        self, model_manager, sample_dataframe
    ):
        """Test that different random states produce different splits."""
        request1 = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        request2 = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=43,
        )

        metrics1 = model_manager.train_model(
            df=sample_dataframe,
            request=request1,
        )
        metrics2 = model_manager.train_model(
            df=sample_dataframe,
            request=request2,
        )

        # Different random states should produce different metrics
        assert metrics1.r2_score != metrics2.r2_score or \
               metrics1.coefficients != metrics2.coefficients

    def test_model_manager_initialization(self, tmp_path):
        """Test model manager initialization."""
        manager = RegressionModelManager(model_dir=str(tmp_path))
        assert manager.model_dir == Path(tmp_path)
        assert manager.model is None
        assert manager.metrics is None

    def test_model_manager_with_custom_model_dir(self, tmp_path):
        """Test model manager with custom directory."""
        custom_dir = tmp_path / "custom_models"
        manager = RegressionModelManager(model_dir=str(custom_dir))
        assert manager.model_dir == custom_dir
        assert custom_dir.exists()  # Directory should be created

    def test_load_nonexistent_model(self, model_manager, tmp_path):
        """Test loading a nonexistent model raises error."""
        nonexistent_path = tmp_path / "nonexistent_model.joblib"
        with pytest.raises(FileNotFoundError):
            model_manager.load_model(str(nonexistent_path))