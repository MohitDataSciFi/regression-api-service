import io
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import joblib
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
def mock_upload_file():
    """Create a mock UploadFile with CSV content."""
    csv_content = b"feature1,feature2,target\n1.0,2.0,3.0\n2.0,3.0,5.0\n3.0,4.0,7.0\n4.0,5.0,9.0\n5.0,6.0,11.0\n"
    return UploadFile(
        filename="test_data.csv",
        file=io.BytesIO(csv_content),
    )


class TestTrainingRequest:
    """Test TrainingRequest validation."""

    def test_valid_request(self):
        """Test valid training request."""
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
        """Test empty target column raises validation error."""
        with pytest.raises(Exception):
            TrainingRequest(target_column="   ", test_size=0.2)

    def test_invalid_test_size_out_of_range(self):
        """Test test_size outside valid range raises validation error."""
        with pytest.raises(Exception):
            TrainingRequest(target_column="target", test_size=0.6)

    def test_invalid_feature_columns_empty_list(self):
        """Test empty feature_columns list raises validation error."""
        with pytest.raises(Exception):
            TrainingRequest(
                target_column="target",
                feature_columns=[],
            )

    def test_invalid_feature_columns_with_empty_string(self):
        """Test feature_columns with empty string raises validation error."""
        with pytest.raises(Exception):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", ""],
            )


class TestModelManager:
    """Test ModelManager core functionality."""

    @pytest.mark.asyncio
    async def test_train_and_save_success(self, model_manager, sample_dataframe):
        """Test successful model training and serialization."""
        # Prepare training data
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        # Create mock upload file
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.csv"
        upload_file.file = io.BytesIO(csv_bytes)

        # Create training request
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        # Mock the file reading
        with patch.object(upload_file, "read", return_value=csv_bytes):
            result = await model_manager.train_and_save(
                file=upload_file,
                request=request,
            )

        # Assert response structure
        assert isinstance(result, TrainingResponse)
        assert result.model_id is not None
        assert "r2" in result.metrics
        assert "mae" in result.metrics
        assert "mse" in result.metrics
        assert "rmse" in result.metrics
        assert "coefficients" in result.diagnostics
        assert "p_values" in result.diagnostics
        assert "feature_importance" in result.feature_importance
        assert result.training_timestamp is not None
        assert result.model_path is not None

        # Verify model file was saved
        model_path = Path(result.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Load and verify model
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict")

    @pytest.mark.asyncio
    async def test_train_and_save_with_feature_columns(self, model_manager, sample_dataframe):
        """Test training with specific feature columns."""
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.csv"
        upload_file.file = io.BytesIO(csv_bytes)

        request = TrainingRequest(
            target_column="target",
            feature_columns=["feature1"],
            test_size=0.2,
            random_state=42,
        )

        with patch.object(upload_file, "read", return_value=csv_bytes):
            result = await model_manager.train_and_save(
                file=upload_file,
                request=request,
            )

        # Verify only feature1 was used
        assert "feature1" in result.feature_importance
        assert "feature2" not in result.feature_importance

    @pytest.mark.asyncio
    async def test_train_and_save_invalid_file_extension(self, model_manager, sample_dataframe):
        """Test training with unsupported file extension."""
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.xyz"
        upload_file.file = io.BytesIO(csv_bytes)

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with patch.object(upload_file, "read", return_value=csv_bytes):
            with pytest.raises(Exception) as exc_info:
                await model_manager.train_and_save(
                    file=upload_file,
                    request=request,
                )
            assert "Unsupported file extension" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_train_and_save_missing_target_column(self, model_manager, sample_dataframe):
        """Test training with missing target column."""
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.csv"
        upload_file.file = io.BytesIO(csv_bytes)

        request = TrainingRequest(
            target_column="nonexistent_column",
            test_size=0.2,
            random_state=42,
        )

        with patch.object(upload_file, "read", return_value=csv_bytes):
            with pytest.raises(Exception) as exc_info:
                await model_manager.train_and_save(
                    file=upload_file,
                    request=request,
                )
            assert "Target column" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_train_and_save_model_diagnostics(self, model_manager, sample_dataframe):
        """Test that model diagnostics are computed correctly."""
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.csv"
        upload_file.file = io.BytesIO(csv_bytes)

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with patch.object(upload_file, "read", return_value=csv_bytes):
            result = await model_manager.train_and_save(
                file=upload_file,
                request=request,
            )

        # Verify diagnostics structure
        assert "r2" in result.metrics
        assert "mae" in result.metrics
        assert "mse" in result.metrics
        assert "rmse" in result.metrics
        assert "coefficients" in result.diagnostics
        assert "p_values" in result.diagnostics
        assert "std_errors" in result.diagnostics
        assert "durbin_watson" in result.diagnostics
        assert "breusch_pagan" in result.diagnostics

        # Verify metrics are numeric and reasonable
        assert 0 <= result.metrics["r2"] <= 1
        assert result.metrics["mae"] >= 0
        assert result.metrics["mse"] >= 0
        assert result.metrics["rmse"] >= 0

        # Verify feature importance
        assert len(result.feature_importance) > 0
        for feature, importance in result.feature_importance.items():
            assert isinstance(feature, str)
            assert isinstance(importance, float)

    @pytest.mark.asyncio
    async def test_train_and_save_file_size_limit(self, model_manager):
        """Test file size limit enforcement."""
        # Create a file larger than MAX_FILE_SIZE
        large_content = b"x" * (MAX_FILE_SIZE + 1)
        
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "large_data.csv"
        upload_file.file = io.BytesIO(large_content)

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with patch.object(upload_file, "read", return_value=large_content):
            with pytest.raises(Exception) as exc_info:
                await model_manager.train_and_save(
                    file=upload_file,
                    request=request,
                )
            assert "File size exceeds" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_model_id_uniqueness(self, model_manager, sample_dataframe):
        """Test that different training runs produce unique model IDs."""
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        upload_file1 = MagicMock(spec=UploadFile)
        upload_file1.filename = "test_data.csv"
        upload_file1.file = io.BytesIO(csv_bytes)

        upload_file2 = MagicMock(spec=UploadFile)
        upload_file2.filename = "test_data.csv"
        upload_file2.file = io.BytesIO(csv_bytes)

        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with patch.object(upload_file1, "read", return_value=csv_bytes):
            result1 = await model_manager.train_and_save(
                file=upload_file1,
                request=request,
            )

        with patch.object(upload_file2, "read", return_value=csv_bytes):
            result2 = await model_manager.train_and_save(
                file=upload_file2,
                request=request,
            )

        assert result1.model_id != result2.model_id