import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile

from src.model_manager import (
    MAX_FILE_SIZE,
    SUPPORTED_EXTENSIONS,
    ModelManager,
    TrainingRequest,
    TrainingResponse,
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
    """Create a ModelManager instance with temporary model directory."""
    manager = ModelManager()
    manager.model_dir = tmp_path
    return manager


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile with CSV content."""
    csv_content = """feature1,feature2,target
1.0,2.0,3.5
2.0,3.0,5.5
3.0,4.0,7.5
4.0,5.0,9.5
5.0,6.0,11.5
"""
    upload_file = MagicMock(spec=UploadFile)
    upload_file.filename = "test_data.csv"
    upload_file.file = io.BytesIO(csv_content.encode())
    upload_file.size = len(csv_content.encode())
    return upload_file


class TestTrainingRequest:
    def test_valid_request(self):
        """Test valid TrainingRequest creation."""
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

    def test_invalid_target_column_empty(self):
        """Test validation error for empty target column."""
        with pytest.raises(ValueError, match="Target column cannot be empty"):
            TrainingRequest(target_column="   ", test_size=0.2)

    def test_invalid_test_size(self):
        """Test validation error for invalid test_size."""
        with pytest.raises(ValueError):
            TrainingRequest(target_column="target", test_size=0.6)

    def test_invalid_feature_columns_duplicates(self):
        """Test validation error for duplicate feature columns."""
        with pytest.raises(ValueError, match="Feature columns must be unique"):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )

    def test_invalid_feature_columns_empty_list(self):
        """Test validation error for empty feature columns list."""
        with pytest.raises(ValueError, match="Feature columns list cannot be empty"):
            TrainingRequest(target_column="target", feature_columns=[])


class TestModelManager:
    def test_initialization(self, model_manager):
        """Test ModelManager initialization."""
        assert model_manager.model_dir is not None
        assert hasattr(model_manager, "_lock")

    def test_train_model_success(self, model_manager, sample_dataframe):
        """Test successful model training."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        result = model_manager.train_model(sample_dataframe, request)

        assert isinstance(result, TrainingResponse)
        assert result.r_squared > 0
        assert result.mse > 0
        assert result.rmse > 0
        assert result.mae > 0
        assert result.training_samples > 0
        assert result.test_samples > 0
        assert result.feature_count == 2
        assert result.model_path.endswith(".joblib")
        assert Path(result.model_path).exists()

    def test_train_model_with_feature_columns(self, model_manager, sample_dataframe):
        """Test training with specific feature columns."""
        request = TrainingRequest(
            target_column="target",
            feature_columns=["feature1"],
            test_size=0.2,
            random_state=42,
        )

        result = model_manager.train_model(sample_dataframe, request)

        assert result.feature_count == 1
        assert "feature1" in result.coefficients
        assert "feature2" not in result.coefficients

    def test_train_model_missing_target(self, model_manager, sample_dataframe):
        """Test training with missing target column."""
        request = TrainingRequest(
            target_column="nonexistent",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            model_manager.train_model(sample_dataframe, request)

    def test_train_model_missing_feature(self, model_manager, sample_dataframe):
        """Test training with missing feature column."""
        request = TrainingRequest(
            target_column="target",
            feature_columns=["nonexistent"],
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Feature column 'nonexistent' not found"):
            model_manager.train_model(sample_dataframe, request)

    def test_train_model_serialization(self, model_manager, sample_dataframe):
        """Test that model is properly serialized."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        result = model_manager.train_model(sample_dataframe, request)

        # Verify model file exists and can be loaded
        model_path = Path(result.model_path)
        assert model_path.exists()
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict")

    def test_train_model_diagnostics(self, model_manager, sample_dataframe):
        """Test that diagnostics are computed correctly."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        result = model_manager.train_model(sample_dataframe, request)

        # Check that p-values are present for each feature
        assert len(result.p_values) == 2
        assert all(0 <= p <= 1 for p in result.p_values.values())

        # Check adjusted R² is less than or equal to R²
        assert result.adjusted_r_squared <= result.r_squared

    def test_validate_upload_file_success(self, model_manager, mock_upload_file):
        """Test successful file validation."""
        result = model_manager.validate_upload_file(mock_upload_file)
        assert result is True

    def test_validate_upload_file_unsupported_extension(self, model_manager, mock_upload_file):
        """Test validation with unsupported file extension."""
        mock_upload_file.filename = "test_data.xlsx"
        with pytest.raises(HTTPException, match="Unsupported file type"):
            model_manager.validate_upload_file(mock_upload_file)

    def test_validate_upload_file_too_large(self, model_manager, mock_upload_file):
        """Test validation with file too large."""
        mock_upload_file.size = MAX_FILE_SIZE + 1
        with pytest.raises(HTTPException, match="File size exceeds"):
            model_manager.validate_upload_file(mock_upload_file)

    def test_parse_csv(self, model_manager, mock_upload_file):
        """Test CSV parsing."""
        df = model_manager.parse_csv(mock_upload_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["feature1", "feature2", "target"]

    def test_parse_csv_invalid_content(self, model_manager):
        """Test CSV parsing with invalid content."""
        mock_upload_file = MagicMock(spec=UploadFile)
        mock_upload_file.filename = "test.csv"
        mock_upload_file.file = io.BytesIO(b"invalid,csv,content\n1,2,3\n4,5")
        
        with pytest.raises(HTTPException, match="Failed to parse CSV"):
            model_manager.parse_csv(mock_upload_file)

    def test_save_model(self, model_manager, sample_dataframe):
        """Test model saving."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        
        # Train a model first
        result = model_manager.train_model(sample_dataframe, request)
        
        # Verify model file was saved
        model_path = Path(result.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"