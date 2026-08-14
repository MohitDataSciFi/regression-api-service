import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile

from src.model_manager import (
    ModelManager,
    TrainingRequest,
    TrainingResponse,
    MODEL_DIR,
    MAX_FILE_SIZE,
    SUPPORTED_EXTENSIONS,
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
def model_manager(tmp_path):
    """Create a ModelManager instance with a temporary directory."""
    return ModelManager(model_dir=tmp_path)


@pytest.fixture
def valid_training_request():
    """Create a valid training request."""
    return TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"],
    )


class TestTrainingRequest:
    """Tests for TrainingRequest validation."""

    def test_valid_request(self):
        """Test that a valid request passes validation."""
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

    def test_empty_target_column_raises_error(self):
        """Test that empty target column raises validation error."""
        with pytest.raises(ValueError, match="Target column cannot be empty"):
            TrainingRequest(
                target_column="   ",
                test_size=0.2,
                random_state=42,
            )

    def test_duplicate_feature_columns_raises_error(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(ValueError, match="Feature columns must be unique"):
            TrainingRequest(
                target_column="target",
                test_size=0.2,
                random_state=42,
                feature_columns=["feature1", "feature1"],
            )

    def test_empty_feature_columns_list_raises_error(self):
        """Test that empty feature columns list raises validation error."""
        with pytest.raises(ValueError, match="Feature columns list cannot be empty"):
            TrainingRequest(
                target_column="target",
                test_size=0.2,
                random_state=42,
                feature_columns=[],
            )

    def test_invalid_test_size_raises_error(self):
        """Test that invalid test size raises validation error."""
        with pytest.raises(ValueError):
            TrainingRequest(
                target_column="target",
                test_size=0.6,  # > 0.5
                random_state=42,
            )


class TestModelManager:
    """Tests for ModelManager class."""

    def test_initialization_creates_directory(self, tmp_path):
        """Test that ModelManager creates the model directory."""
        model_dir = tmp_path / "test_models"
        manager = ModelManager(model_dir=model_dir)
        assert model_dir.exists()
        assert model_dir.is_dir()

    def test_train_model_returns_valid_response(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that train_model returns a valid TrainingResponse."""
        # Convert dataframe to CSV bytes for upload simulation
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        # Create a mock UploadFile
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test_data.csv"
        mock_file.content_type = "text/csv"
        mock_file.file = io.BytesIO(csv_bytes)

        # Train the model
        response = model_manager.train_model(mock_file, valid_training_request)

        # Assert response is valid
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert response.training_timestamp is not None
        assert response.r_squared > 0
        assert response.adjusted_r_squared > 0
        assert response.mse > 0
        assert response.rmse > 0
        assert response.mae > 0
        assert response.feature_count == 2
        assert response.sample_count == 100
        assert response.model_path.endswith(".joblib")
        assert "breusch_pagan" in response.diagnostics
        assert "durbin_watson" in response.diagnostics

    def test_train_model_saves_model_file(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that train_model saves the model to disk."""
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test_data.csv"
        mock_file.content_type = "text/csv"
        mock_file.file = io.BytesIO(csv_bytes)

        response = model_manager.train_model(mock_file, valid_training_request)

        # Check that model file exists
        model_path = Path(response.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Load and verify the model
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict")
        assert hasattr(loaded_model, "coef_")

    def test_train_model_with_invalid_file_extension(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training with invalid file extension raises error."""
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test_data.pdf"  # Invalid extension
        mock_file.content_type = "application/pdf"
        mock_file.file = io.BytesIO(csv_bytes)

        with pytest.raises(ValueError, match="Unsupported file type"):
            model_manager.train_model(mock_file, valid_training_request)

    def test_train_model_with_missing_target_column(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training with missing target column raises error."""
        # Remove target column from dataframe
        df_without_target = sample_dataframe.drop(columns=["target"])
        csv_buffer = io.StringIO()
        df_without_target.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test_data.csv"
        mock_file.content_type = "text/csv"
        mock_file.file = io.BytesIO(csv_bytes)

        with pytest.raises(ValueError, match="Target column 'target' not found"):
            model_manager.train_model(mock_file, valid_training_request)

    def test_train_model_with_missing_feature_column(
        self, model_manager, sample_dataframe
    ):
        """Test that training with missing feature column raises error."""
        # Create request with non-existent feature column
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "nonexistent_feature"],
        )

        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test_data.csv"
        mock_file.content_type = "text/csv"
        mock_file.file = io.BytesIO(csv_bytes)

        with pytest.raises(ValueError, match="Feature column 'nonexistent_feature' not found"):
            model_manager.train_model(mock_file, request)

    def test_train_model_with_large_file(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training with file larger than MAX_FILE_SIZE raises error."""
        # Create a large dataframe
        large_df = pd.concat([sample_dataframe] * 1000, ignore_index=True)
        csv_buffer = io.StringIO()
        large_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        # Ensure the file is larger than MAX_FILE_SIZE
        assert len(csv_bytes) > MAX_FILE_SIZE

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test_data.csv"
        mock_file.content_type = "text/csv"
        mock_file.file = io.BytesIO(csv_bytes)

        with pytest.raises(ValueError, match="File size exceeds maximum allowed"):
            model_manager.train_model(mock_file, valid_training_request)

    def test_load_model(self, model_manager, sample_dataframe, valid_training_request):
        """Test that load_model loads a previously saved model."""
        # First train and save a model
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test_data.csv"
        mock_file.content_type = "text/csv"
        mock_file.file = io.BytesIO(csv_bytes)

        response = model_manager.train_model(mock_file, valid_training_request)

        # Load the model
        loaded_model = model_manager.load_model(response.model_id)

        # Verify the loaded model works
        assert hasattr(loaded_model, "predict")
        test_features = sample_dataframe[["feature1", "feature2"]].iloc[:5]
        predictions = loaded_model.predict(test_features)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    def test_load_nonexistent_model(self, model_manager):
        """Test that loading a non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            model_manager.load_model("nonexistent_model_id")