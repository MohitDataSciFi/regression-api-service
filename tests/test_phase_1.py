import asyncio
import io
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from src.model_manager import (
    ErrorResponse,
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
    return ModelManager(model_dir=str(tmp_path / "models"))


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

    def test_valid_request(self, valid_training_request):
        """Test that a valid request passes validation."""
        assert valid_training_request.target_column == "target"
        assert valid_training_request.test_size == 0.2
        assert valid_training_request.random_state == 42
        assert valid_training_request.feature_columns == ["feature1", "feature2"]

    def test_empty_target_column(self):
        """Test that empty target column raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="   ", test_size=0.2)

    def test_invalid_test_size(self):
        """Test that test_size outside [0.1, 0.5] raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", test_size=0.6)
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", test_size=0.05)

    def test_duplicate_feature_columns(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )

    def test_empty_feature_columns(self):
        """Test that empty feature columns list raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", feature_columns=[])


class TestModelManager:
    """Tests for ModelManager core functionality."""

    @pytest.mark.asyncio
    async def test_train_success(self, model_manager, sample_dataframe, valid_training_request):
        """Test successful model training with valid data."""
        # Convert dataframe to CSV bytes for upload
        csv_bytes = sample_dataframe.to_csv(index=False).encode()
        upload_file = UploadFile(
            filename="data.csv",
            file=io.BytesIO(csv_bytes),
        )

        response = await model_manager.train(
            file=upload_file,
            request=valid_training_request,
        )

        # Verify response structure
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert response.r_squared > 0.5  # Should have decent R² with this data
        assert response.training_samples == 80  # 80% of 100
        assert response.test_samples == 20  # 20% of 100
        assert response.feature_columns == ["feature1", "feature2"]
        assert response.target_column == "target"
        assert response.created_at is not None
        assert response.model_path is not None

        # Verify model file was saved
        model_path = Path(response.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Verify metrics are reasonable
        assert 0 <= response.mse <= 10
        assert 0 <= response.mae <= 5
        assert 0 <= response.rmse <= 5
        assert len(response.coefficients) == 2
        assert len(response.p_values) == 2

    @pytest.mark.asyncio
    async def test_train_without_feature_columns(self, model_manager, sample_dataframe):
        """Test training when feature_columns is None (uses all except target)."""
        csv_bytes = sample_dataframe.to_csv(index=False).encode()
        upload_file = UploadFile(
            filename="data.csv",
            file=io.BytesIO(csv_bytes),
        )
        request = TrainingRequest(target_column="target", test_size=0.2)

        response = await model_manager.train(file=upload_file, request=request)

        assert response.feature_columns == ["feature1", "feature2"]
        assert response.training_samples == 80

    @pytest.mark.asyncio
    async def test_train_invalid_csv(self, model_manager, valid_training_request):
        """Test training with invalid CSV data."""
        upload_file = UploadFile(
            filename="data.csv",
            file=io.BytesIO(b"invalid,csv,data\n1,2,3\n4,5"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await model_manager.train(file=upload_file, request=valid_training_request)

        assert exc_info.value.status_code == 400
        assert "error" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_train_missing_target_column(self, model_manager, sample_dataframe, valid_training_request):
        """Test training when target column doesn't exist in data."""
        # Remove target column from data
        df_without_target = sample_dataframe.drop(columns=["target"])
        csv_bytes = df_without_target.to_csv(index=False).encode()
        upload_file = UploadFile(
            filename="data.csv",
            file=io.BytesIO(csv_bytes),
        )

        with pytest.raises(HTTPException) as exc_info:
            await model_manager.train(file=upload_file, request=valid_training_request)

        assert exc_info.value.status_code == 400
        assert "target" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_train_missing_feature_column(self, model_manager, sample_dataframe, valid_training_request):
        """Test training when a specified feature column doesn't exist."""
        # Create request with non-existent feature
        bad_request = TrainingRequest(
            target_column="target",
            feature_columns=["feature1", "nonexistent_feature"],
        )
        csv_bytes = sample_dataframe.to_csv(index=False).encode()
        upload_file = UploadFile(
            filename="data.csv",
            file=io.BytesIO(csv_bytes),
        )

        with pytest.raises(HTTPException) as exc_info:
            await model_manager.train(file=upload_file, request=bad_request)

        assert exc_info.value.status_code == 400
        assert "feature" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_model_serialization(self, model_manager, sample_dataframe, valid_training_request):
        """Test that the trained model can be loaded back from disk."""
        csv_bytes = sample_dataframe.to_csv(index=False).encode()
        upload_file = UploadFile(
            filename="data.csv",
            file=io.BytesIO(csv_bytes),
        )

        response = await model_manager.train(file=upload_file, request=valid_training_request)

        # Load the saved model
        loaded_model = joblib.load(response.model_path)
        assert loaded_model is not None

        # Verify the model can make predictions
        test_data = sample_dataframe[["feature1", "feature2"]].iloc[:5]
        predictions = loaded_model.predict(test_data)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    @pytest.mark.asyncio
    async def test_concurrent_training(self, model_manager, sample_dataframe, valid_training_request):
        """Test that concurrent training requests are handled safely."""
        csv_bytes = sample_dataframe.to_csv(index=False).encode()
        upload_file1 = UploadFile(
            filename="data1.csv",
            file=io.BytesIO(csv_bytes),
        )
        upload_file2 = UploadFile(
            filename="data2.csv",
            file=io.BytesIO(csv_bytes),
        )

        # Run two training requests concurrently
        responses = await asyncio.gather(
            model_manager.train(file=upload_file1, request=valid_training_request),
            model_manager.train(file=upload_file2, request=valid_training_request),
        )

        # Both should succeed with different model IDs
        assert responses[0].model_id != responses[1].model_id
        assert responses[0].model_path != responses[1].model_path
        assert responses[0].r_squared == responses[1].r_squared  # Same data, same result

    @pytest.mark.asyncio
    async def test_error_response_model(self):
        """Test ErrorResponse model validation."""
        error_response = ErrorResponse(
            error="Test error",
            detail="Test detail",
            timestamp=datetime.now(),
        )
        assert error_response.error == "Test error"
        assert error_response.detail == "Test detail"
        assert error_response.timestamp is not None

    @pytest.mark.asyncio
    async def test_training_response_model(self):
        """Test TrainingResponse model validation."""
        response = TrainingResponse(
            model_id="test_model",
            metrics={"r2": 0.95},
            coefficients={"feature1": 1.5},
            p_values={"feature1": 0.01},
            r_squared=0.95,
            adjusted_r_squared=0.94,
            mse=0.5,
            mae=0.4,
            rmse=0.7,
            training_samples=80,
            test_samples=20,
            feature_columns=["feature1"],
            target_column="target",
            created_at=datetime.now(),
            model_path="/tmp/test_model.joblib",
        )
        assert response.model_id == "test_model"
        assert response.r_squared == 0.95