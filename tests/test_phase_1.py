import io
import logging
import time
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
    ModelArtifacts,
    ModelManager,
    TrainingRequest,
    TrainingResponse,
    train_model,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 2 + 1,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_bytes(sample_dataframe):
    """Convert sample dataframe to CSV bytes."""
    return sample_dataframe.to_csv(index=False).encode()


@pytest.fixture
def valid_training_request():
    """Create a valid training request."""
    return TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2", "feature3"],
    )


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with temp directory."""
    return ModelManager(model_dir=str(tmp_path))


class TestTrainingRequestValidation:
    """Test TrainingRequest Pydantic validation."""

    def test_valid_request(self, valid_training_request):
        """Test that a valid request passes validation."""
        assert valid_training_request.target_column == "target"
        assert valid_training_request.test_size == 0.2
        assert valid_training_request.random_state == 42
        assert valid_training_request.feature_columns == ["feature1", "feature2", "feature3"]

    def test_invalid_target_column_empty(self):
        """Test that empty target column raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="   ", test_size=0.2)

    def test_invalid_test_size_out_of_range(self):
        """Test that test_size outside [0.1, 0.5] raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", test_size=0.05)
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", test_size=0.6)

    def test_invalid_feature_columns_empty(self):
        """Test that empty feature_columns list raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                feature_columns=[],
            )

    def test_invalid_feature_columns_duplicates(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )


class TestModelManagerTraining:
    """Test ModelManager training functionality."""

    def test_train_model_success(self, model_manager, sample_dataframe, valid_training_request):
        """Test successful model training with valid data."""
        # Train the model
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify response structure
        assert isinstance(response, TrainingResponse)
        assert response.model_id.startswith("model_")
        assert response.training_timestamp is not None
        assert response.r_squared > 0
        assert response.adjusted_r_squared > 0
        assert len(response.coefficients) == 3  # 3 features
        assert len(response.p_values) == 3
        assert len(response.feature_importance) == 3
        assert response.data_shape == {"rows": 100, "columns": 4}
        assert response.training_duration_seconds > 0

        # Verify model file was saved
        model_path = Path(response.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Verify model can be loaded
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict")

    def test_train_model_without_feature_columns(self, model_manager, sample_dataframe):
        """Test training when feature_columns is None (use all except target)."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=None,
        )

        response = model_manager.train(df=sample_dataframe, request=request)

        # Should use all 3 features
        assert len(response.coefficients) == 3
        assert len(response.p_values) == 3

    def test_train_model_missing_target_column(self, model_manager, sample_dataframe):
        """Test training with missing target column raises error."""
        request = TrainingRequest(
            target_column="nonexistent_column",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Target column 'nonexistent_column' not found"):
            model_manager.train(df=sample_dataframe, request=request)

    def test_train_model_missing_feature_column(self, model_manager, sample_dataframe):
        """Test training with missing feature column raises error."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "nonexistent_feature"],
        )

        with pytest.raises(ValueError, match="Feature column 'nonexistent_feature' not found"):
            model_manager.train(df=sample_dataframe, request=request)

    def test_train_model_insufficient_data(self, model_manager):
        """Test training with insufficient data raises error."""
        # Create tiny dataframe
        small_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "target": [1, 2, 3],
        })

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            model_manager.train(df=small_df, request=request)


class TestModelSerialization:
    """Test model serialization and loading."""

    def test_model_artifacts_serialization(self, model_manager, sample_dataframe, valid_training_request):
        """Test that model artifacts are properly serialized."""
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify model file exists and is valid joblib
        model_path = Path(response.model_path)
        assert model_path.exists()
        
        # Load and verify model
        loaded_model = joblib.load(model_path)
        assert isinstance(loaded_model, LinearRegression)
        
        # Test prediction with loaded model
        test_data = sample_dataframe[valid_training_request.feature_columns].iloc[:5]
        predictions = loaded_model.predict(test_data)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    def test_model_artifacts_metadata(self, model_manager, sample_dataframe, valid_training_request):
        """Test that model artifacts contain correct metadata."""
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify model_id format
        assert response.model_id.startswith("model_")
        
        # Verify timestamp format
        timestamp = datetime.fromisoformat(response.training_timestamp)
        assert timestamp is not None

        # Verify metrics are finite
        for metric_name, metric_value in response.metrics.items():
            assert np.isfinite(metric_value), f"Metric {metric_name} is not finite"

        # Verify coefficients and p-values match
        assert set(response.coefficients.keys()) == set(response.p_values.keys())
        assert set(response.coefficients.keys()) == set(valid_training_request.feature_columns)


class TestTrainingEndpoint:
    """Test the training endpoint functionality."""

    @pytest.mark.asyncio
    async def test_training_endpoint_with_csv(self, model_manager, sample_csv_bytes, valid_training_request):
        """Test training endpoint with CSV file upload."""
        # Create mock UploadFile
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.csv"
        upload_file.content_type = "text/csv"
        upload_file.file = io.BytesIO(sample_csv_bytes)

        # Mock the read method
        async def mock_read():
            return sample_csv_bytes
        upload_file.read = mock_read

        # Call the training function
        response = await train_model(
            file=upload_file,
            request=valid_training_request,
            model_manager=model_manager,
        )

        # Verify response
        assert isinstance(response, TrainingResponse)
        assert response.r_squared > 0
        assert response.model_path is not None

    @pytest.mark.asyncio
    async def test_training_endpoint_invalid_csv(self, model_manager, valid_training_request):
        """Test training endpoint with invalid CSV data."""
        # Create mock UploadFile with invalid data
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "invalid.csv"
        upload_file.content_type = "text/csv"

        invalid_csv = b"not,a,valid,csv\n1,2,3"
        async def mock_read():
            return invalid_csv
        upload_file.read = mock_read

        # Should raise an error for invalid CSV
        with pytest.raises(Exception):
            await train_model(
                file=upload_file,
                request=valid_training_request,
                model_manager=model_manager,
            )

    @pytest.mark.asyncio
    async def test_training_endpoint_missing_target(self, model_manager, sample_csv_bytes):
        """Test training endpoint with missing target column."""
        # Create request with non-existent target
        request = TrainingRequest(
            target_column="nonexistent",
            test_size=0.2,
            random_state=42,
        )

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.csv"
        upload_file.content_type = "text/csv"

        async def mock_read():
            return sample_csv_bytes
        upload_file.read = mock_read

        # Should raise ValueError for missing target
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            await train_model(
                file=upload_file,
                request=request,
                model_manager=model_manager,
            )


class TestModelManagerEdgeCases:
    """Test edge cases in model manager."""

    def test_train_model_with_nan_values(self, model_manager):
        """Test training with NaN values in data."""
        # Create dataframe with NaN values
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "feature2": [1.0, 2.0, 3.0, np.nan, 5.0],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        # Should either handle NaN or raise appropriate error
        try:
            response = model_manager.train(df=df, request=request)
            # If successful, verify response
            assert isinstance(response, TrainingResponse)
        except ValueError as e:
            # If error, it should be about NaN values
            assert "NaN" in str(e) or "missing" in str(e).lower()

    def test_train_model_with_categorical_features(self, model_manager):
        """Test training with categorical features."""
        df = pd.DataFrame({
            "categorical": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["categorical", "numeric"],
        )

        # Should either handle categorical or raise appropriate error
        try:
            response = model_manager.train(df=df, request=request)
            assert isinstance(response, TrainingResponse)
        except (ValueError, TypeError) as e:
            # If error, it should be about categorical data
            assert "categorical" in str(e).lower() or "string" in str(e).lower()

    def test_model_reproducibility(self, model_manager, sample_dataframe, valid_training_request):
        """Test that training is reproducible with same random state."""
        # Train two models with same parameters
        response1 = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )
        response2 = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify same metrics
        assert response1.r_squared == pytest.approx(response2.r_squared)
        assert response1.metrics == pytest.approx(response2.metrics)
        assert response1.coefficients == pytest.approx(response2.coefficients)
        assert response1.p_values == pytest.approx(response2.p_values)