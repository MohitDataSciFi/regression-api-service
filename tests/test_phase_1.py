import io
import logging
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
    train_model,
    validate_and_prepare_data,
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
def sample_csv_bytes():
    """Create sample CSV data as bytes."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 2 + 1,
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode()


@pytest.fixture
def mock_upload_file(sample_csv_bytes):
    """Create a mock UploadFile object."""
    upload_file = MagicMock(spec=UploadFile)
    upload_file.filename = "test_data.csv"
    upload_file.file = io.BytesIO(sample_csv_bytes)
    return upload_file


class TestTrainingRequest:
    def test_valid_training_request(self):
        """Test that a valid TrainingRequest is created successfully."""
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
        """Test that test_size validation fails for out-of-range values."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", test_size=0.6)

    def test_empty_feature_columns(self):
        """Test that empty feature_columns list raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", feature_columns=[])

    def test_duplicate_feature_columns(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )


class TestValidateAndPrepareData:
    def test_valid_data_preparation(self, sample_dataframe):
        """Test that valid data is prepared correctly."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test, feature_names = validate_and_prepare_data(
            sample_dataframe, request
        )
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
        assert feature_names == ["feature1", "feature2"]

    def test_missing_target_column(self, sample_dataframe):
        """Test that missing target column raises DataValidationError."""
        request = TrainingRequest(target_column="nonexistent")
        
        with pytest.raises(DataValidationError):
            validate_and_prepare_data(sample_dataframe, request)

    def test_specific_feature_columns(self, sample_dataframe):
        """Test that specific feature columns are used correctly."""
        request = TrainingRequest(
            target_column="target",
            feature_columns=["feature1"],
        )
        X_train, X_test, y_train, y_test, feature_names = validate_and_prepare_data(
            sample_dataframe, request
        )
        
        assert feature_names == ["feature1"]
        assert X_train.shape[1] == 1

    def test_non_numeric_data(self):
        """Test that non-numeric data raises DataValidationError."""
        df = pd.DataFrame({
            "feature1": ["a", "b", "c"],
            "target": [1, 2, 3],
        })
        request = TrainingRequest(target_column="target")
        
        with pytest.raises(DataValidationError):
            validate_and_prepare_data(df, request)


class TestTrainModel:
    def test_successful_training(self, sample_dataframe, tmp_path):
        """Test that model training succeeds and returns expected metrics."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        
        with patch("src.model_manager.joblib.dump") as mock_dump:
            result = train_model(
                sample_dataframe,
                request,
                model_path=str(tmp_path / "model.joblib"),
            )
        
        # Verify model was saved
        mock_dump.assert_called_once()
        
        # Verify metrics are present
        assert "r2_score" in result
        assert "rmse" in result
        assert "mae" in result
        assert "coefficients" in result
        assert "p_values" in result
        assert "training_samples" in result
        assert "test_samples" in result
        
        # Verify metric values are reasonable
        assert 0 <= result["r2_score"] <= 1
        assert result["rmse"] >= 0
        assert result["mae"] >= 0
        assert len(result["coefficients"]) == 2  # feature1, feature2
        assert len(result["p_values"]) == 2

    def test_training_with_specific_features(self, sample_dataframe, tmp_path):
        """Test training with specific feature columns."""
        request = TrainingRequest(
            target_column="target",
            feature_columns=["feature1"],
        )
        
        with patch("src.model_manager.joblib.dump"):
            result = train_model(
                sample_dataframe,
                request,
                model_path=str(tmp_path / "model.joblib"),
            )
        
        assert len(result["coefficients"]) == 1
        assert len(result["p_values"]) == 1

    def test_training_error_handling(self, sample_dataframe, tmp_path):
        """Test that training errors are properly wrapped."""
        request = TrainingRequest(target_column="target")
        
        with patch(
            "src.model_manager.LinearRegression.fit",
            side_effect=Exception("Training failed"),
        ):
            with pytest.raises(ModelTrainingError):
                train_model(
                    sample_dataframe,
                    request,
                    model_path=str(tmp_path / "model.joblib"),
                )

    def test_model_serialization(self, sample_dataframe, tmp_path):
        """Test that the model is properly serialized."""
        request = TrainingRequest(target_column="target")
        model_path = str(tmp_path / "test_model.joblib")
        
        with patch("src.model_manager.joblib.dump") as mock_dump:
            train_model(sample_dataframe, request, model_path=model_path)
        
        # Verify joblib.dump was called with correct arguments
        mock_dump.assert_called_once()
        args, kwargs = mock_dump.call_args
        assert args[1] == model_path  # model path is second argument
        assert "model" in args[0]  # first argument contains model dict
        assert "metadata" in args[0]
        assert args[0]["metadata"]["target_column"] == "target"
        assert args[0]["metadata"]["feature_columns"] == ["feature1", "feature2"]