import io
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from src.model_manager import (
    DataValidationError,
    ModelTrainingError,
    TrainingRequest,
    TrainingResponse,
    logger,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 2 + 1,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_bytes(sample_dataframe):
    """Convert sample dataframe to CSV bytes."""
    return sample_dataframe.to_csv(index=False).encode()


@pytest.fixture
def mock_upload_file(sample_csv_bytes):
    """Create a mock UploadFile."""
    upload_file = MagicMock(spec=UploadFile)
    upload_file.filename = "test_data.csv"
    upload_file.file = io.BytesIO(sample_csv_bytes)
    return upload_file


class TestTrainingRequest:
    """Tests for TrainingRequest validation."""

    def test_valid_request(self):
        """Test valid training request passes validation."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "feature2"],
        )
        assert request.target_column == "target"
        assert request.test_size == 0.2
        assert request.random_state == 42
        assert request.feature_columns == ["feature1", "feature2"]

    def test_invalid_test_size(self):
        """Test invalid test_size raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                test_size=0.6,  # > 0.5
            )

    def test_empty_feature_columns(self):
        """Test empty feature columns raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                feature_columns=[],  # empty list
            )

    def test_duplicate_feature_columns_removed(self):
        """Test duplicate feature columns are removed."""
        request = TrainingRequest(
            target_column="target",
            feature_columns=["feature1", "feature1", "feature2"],
        )
        assert request.feature_columns == ["feature1", "feature2"]


class TestModelTraining:
    """Tests for model training functionality."""

    @patch("src.model_manager.joblib.dump")
    def test_model_training_success(self, mock_dump, sample_dataframe, tmp_path):
        """Test successful model training and serialization."""
        from src.model_manager import ModelManager

        # Mock the model manager's training method
        manager = ModelManager()
        model_path = tmp_path / "test_model.joblib"

        # Train model
        result = manager.train(
            data=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
            model_path=model_path,
        )

        # Assert training results
        assert result["model_id"] is not None
        assert result["training_timestamp"] is not None
        assert "metrics" in result
        assert "coefficients" in result
        assert "r2" in result["metrics"]
        assert "mae" in result["metrics"]
        assert "mse" in result["metrics"]

        # Verify model was saved
        mock_dump.assert_called_once()

    @patch("src.model_manager.joblib.dump")
    def test_model_training_with_missing_target(self, mock_dump, sample_dataframe, tmp_path):
        """Test training with missing target column raises error."""
        from src.model_manager import ModelManager

        manager = ModelManager()
        model_path = tmp_path / "test_model.joblib"

        # Remove target column
        df_without_target = sample_dataframe.drop(columns=["target"])

        with pytest.raises(DataValidationError):
            manager.train(
                data=df_without_target,
                target_column="target",
                feature_columns=["feature1", "feature2"],
                test_size=0.2,
                random_state=42,
                model_path=model_path,
            )

    @patch("src.model_manager.joblib.dump")
    def test_model_training_with_invalid_features(self, mock_dump, sample_dataframe, tmp_path):
        """Test training with invalid feature columns raises error."""
        from src.model_manager import ModelManager

        manager = ModelManager()
        model_path = tmp_path / "test_model.joblib"

        # Use non-existent feature column
        with pytest.raises(DataValidationError):
            manager.train(
                data=sample_dataframe,
                target_column="target",
                feature_columns=["non_existent_feature"],
                test_size=0.2,
                random_state=42,
                model_path=model_path,
            )

    def test_model_serialization_roundtrip(self, sample_dataframe, tmp_path):
        """Test model serialization and deserialization roundtrip."""
        from src.model_manager import ModelManager

        manager = ModelManager()
        model_path = tmp_path / "test_model.joblib"

        # Train and save model
        result = manager.train(
            data=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
            model_path=model_path,
        )

        # Load model back
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

        # Verify model can make predictions
        X_test = sample_dataframe[["feature1", "feature2"]].iloc[:5]
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))


class TestTrainingResponse:
    """Tests for TrainingResponse model."""

    def test_training_response_creation(self):
        """Test TrainingResponse can be created with valid data."""
        response = TrainingResponse(
            model_id="test_model_123",
            training_timestamp=datetime.now(),
            metrics={"r2": 0.85, "mae": 0.5, "mse": 0.4},
            coefficients={"feature1": 0.5, "feature2": -0.3},
        )
        assert response.model_id == "test_model_123"
        assert response.metrics["r2"] == 0.85
        assert response.coefficients["feature1"] == 0.5

    def test_training_response_invalid_metrics(self):
        """Test TrainingResponse with invalid metrics type."""
        with pytest.raises(ValidationError):
            TrainingResponse(
                model_id="test_model_123",
                training_timestamp=datetime.now(),
                metrics={"r2": "not_a_number"},  # Invalid type
                coefficients={"feature1": 0.5},
            )


class TestLogging:
    """Tests for logging configuration."""

    def test_logger_configured(self):
        """Test logger has handlers configured."""
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 2  # Console and file handlers

    def test_log_file_created(self, tmp_path):
        """Test log file is created when logging."""
        log_file = tmp_path / "test_log.log"
        test_logger = logging.getLogger("test_logger")
        test_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        test_logger.addHandler(handler)
        
        test_logger.info("Test log message")
        handler.flush()
        
        assert log_file.exists()
        assert log_file.stat().st_size > 0