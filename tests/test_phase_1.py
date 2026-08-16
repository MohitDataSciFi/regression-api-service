import io
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile

from src.model_manager import (
    MAX_FILE_SIZE,
    MODEL_DIR,
    SUPPORTED_EXTENSIONS,
    ModelManager,
    TrainingRequest,
    TrainingResponse,
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
    """Test TrainingRequest validation."""

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

    def test_invalid_target_column_empty(self):
        """Test that empty target column raises validation error."""
        with pytest.raises(ValueError, match="Target column cannot be empty"):
            TrainingRequest(target_column="   ", test_size=0.2)

    def test_invalid_test_size_bounds(self):
        """Test that test_size outside bounds raises validation error."""
        with pytest.raises(ValueError):
            TrainingRequest(target_column="target", test_size=0.6)
        with pytest.raises(ValueError):
            TrainingRequest(target_column="target", test_size=0.05)

    def test_invalid_feature_columns_empty_list(self):
        """Test that empty feature_columns list raises validation error."""
        with pytest.raises(ValueError, match="Feature columns list cannot be empty"):
            TrainingRequest(
                target_column="target",
                feature_columns=[],
            )

    def test_invalid_feature_columns_duplicates(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(ValueError, match="Feature columns must be unique"):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )


class TestModelManagerTraining:
    """Test ModelManager training functionality."""

    def test_train_model_success(self, model_manager, sample_dataframe, valid_training_request):
        """Test successful model training with valid data."""
        # Train the model
        response = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify response structure
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert response.r_squared > 0
        assert response.adjusted_r_squared > 0
        assert response.mse > 0
        assert response.rmse > 0
        assert response.mae > 0
        assert response.training_samples > 0
        assert response.test_samples > 0
        assert response.feature_count == 2
        assert response.training_time > 0
        assert response.created_at is not None
        assert response.model_path is not None

        # Verify coefficients and p-values
        assert len(response.coefficients) == 3  # 2 features + intercept
        assert len(response.p_values) == 3
        assert all(isinstance(v, float) for v in response.coefficients.values())
        assert all(isinstance(v, float) for v in response.p_values.values())

        # Verify model file was saved
        model_path = Path(response.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Verify model can be loaded
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

    def test_train_model_without_feature_columns(self, model_manager, sample_dataframe):
        """Test training when feature_columns is None (use all other columns)."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        response = model_manager.train_model(
            df=sample_dataframe,
            request=request,
        )

        assert response.feature_count == 2  # feature1 and feature2
        assert len(response.coefficients) == 3

    def test_train_model_missing_target_column(self, model_manager, sample_dataframe):
        """Test training with non-existent target column."""
        request = TrainingRequest(
            target_column="nonexistent_column",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Target column 'nonexistent_column' not found"):
            model_manager.train_model(
                df=sample_dataframe,
                request=request,
            )

    def test_train_model_missing_feature_column(self, model_manager, sample_dataframe):
        """Test training with non-existent feature column."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["nonexistent_feature"],
        )

        with pytest.raises(ValueError, match="Feature column 'nonexistent_feature' not found"):
            model_manager.train_model(
                df=sample_dataframe,
                request=request,
            )

    def test_train_model_insufficient_data(self, model_manager):
        """Test training with insufficient data."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "target": [1, 2, 3],
        })
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            model_manager.train_model(
                df=df,
                request=request,
            )

    def test_train_model_with_nan_values(self, model_manager, sample_dataframe):
        """Test training with NaN values in data."""
        df = sample_dataframe.copy()
        df.loc[0, "feature1"] = np.nan
        df.loc[1, "target"] = np.nan

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="NaN values"):
            model_manager.train_model(
                df=df,
                request=request,
            )

    def test_train_model_serialization(self, model_manager, sample_dataframe, valid_training_request):
        """Test that model is properly serialized and can be loaded."""
        response = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Load the model and verify it works
        loaded_model = joblib.load(response.model_path)
        assert loaded_model is not None

        # Make a prediction with the loaded model
        test_data = sample_dataframe[valid_training_request.feature_columns].iloc[:5]
        predictions = loaded_model.predict(test_data)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))


class TestModelManagerFileHandling:
    """Test file handling and validation."""

    def test_validate_file_size(self, model_manager):
        """Test file size validation."""
        # Create a mock upload file
        mock_file = MagicMock(spec=UploadFile)
        mock_file.size = MAX_FILE_SIZE + 1

        with pytest.raises(HTTPException, match="File size exceeds"):
            model_manager.validate_file(mock_file)

    def test_validate_file_extension(self, model_manager):
        """Test file extension validation."""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.size = 1000
        mock_file.filename = "data.pdf"

        with pytest.raises(HTTPException, match="Unsupported file type"):
            model_manager.validate_file(mock_file)

    def test_validate_file_success(self, model_manager):
        """Test valid file passes validation."""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.size = 1000
        mock_file.filename = "data.csv"

        # Should not raise any exception
        model_manager.validate_file(mock_file)

    def test_parse_csv_file(self, model_manager, sample_dataframe):
        """Test CSV file parsing."""
        # Create a CSV file in memory
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "data.csv"
        mock_file.read = MagicMock(return_value=csv_bytes)

        df = model_manager.parse_file(mock_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_parse_txt_file(self, model_manager, sample_dataframe):
        """Test TXT file parsing."""
        # Create a TXT file in memory
        txt_buffer = io.StringIO()
        sample_dataframe.to_csv(txt_buffer, index=False, sep="\t")
        txt_bytes = txt_buffer.getvalue().encode()

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "data.txt"
        mock_file.read = MagicMock(return_value=txt_bytes)

        df = model_manager.parse_file(mock_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_parse_invalid_file(self, model_manager):
        """Test parsing invalid file content."""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "data.csv"
        mock_file.read = MagicMock(return_value=b"invalid,csv,content\n1,2,3\n4,5")

        with pytest.raises(HTTPException, match="Failed to parse"):
            model_manager.parse_file(mock_file)


class TestModelManagerDiagnostics:
    """Test model diagnostics computation."""

    def test_compute_diagnostics(self, model_manager, sample_dataframe, valid_training_request):
        """Test diagnostic metrics computation."""
        response = model_manager.train_model(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify all diagnostic metrics are present and valid
        assert response.r_squared >= 0 and response.r_squared <= 1
        assert response.adjusted_r_squared >= 0 and response.adjusted_r_squared <= 1
        assert response.mse >= 0
        assert response.rmse >= 0
        assert response.mae >= 0

        # Verify p-values are between 0 and 1
        for p_value in response.p_values.values():
            assert 0 <= p_value <= 1

        # Verify coefficients are finite
        for coefficient in response.coefficients.values():
            assert np.isfinite(coefficient)

    def test_diagnostics_with_perfect_fit(self, model_manager):
        """Test diagnostics with perfect linear relationship."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X[:, 0] + 1

        df = pd.DataFrame({
            "feature1": X[:, 0],
            "target": y,
        })

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1"],
        )

        response = model_manager.train_model(
            df=df,
            request=request,
        )

        # Perfect fit should have R² close to 1
        assert response.r_squared > 0.99
        assert response.mse < 0.01


class TestModelManagerIntegration:
    """Test integration with FastAPI endpoints."""

    def test_training_endpoint(self, model_manager, sample_dataframe):
        """Test the training endpoint with mock FastAPI."""
        from fastapi.testclient import TestClient
        from src.main import app

        # Override the model manager dependency
        app.dependency_overrides[get_model_manager] = lambda: model_manager

        client = TestClient(app)

        # Create a CSV file for upload
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        # Make the request
        response = client.post(
            "/train",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            data={
                "target_column": "target",
                "test_size": "0.2",
                "random_state": "42",
                "feature_columns": '["feature1", "feature2"]',
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "metrics" in data
        assert "coefficients" in data
        assert "p_values" in data
        assert "r_squared" in data

        app.dependency_overrides.clear()

    def test_training_endpoint_invalid_request(self, model_manager, sample_dataframe):
        """Test training endpoint with invalid request."""
        from fastapi.testclient import TestClient
        from src.main import app

        app.dependency_overrides[get_model_manager] = lambda: model_manager

        client = TestClient(app)

        # Create a CSV file for upload
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        # Make request with invalid test_size
        response = client.post(
            "/train",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            data={
                "target_column": "target",
                "test_size": "0.8",  # Invalid: > 0.5
                "random_state": "42",
            },
        )

        assert response.status_code == 422  # Validation error

        app.dependency_overrides.clear()