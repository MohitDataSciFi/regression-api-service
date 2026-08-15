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
    MODEL_DIR,
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
def sample_csv_bytes(sample_dataframe):
    """Create CSV bytes from sample dataframe."""
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode()


class TestTrainingRequest:
    def test_valid_request(self):
        """Test that a valid TrainingRequest passes validation."""
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

    def test_invalid_target_column(self):
        """Test that empty target column raises validation error."""
        with pytest.raises(Exception):
            TrainingRequest(target_column="   ", test_size=0.2)

    def test_duplicate_feature_columns(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(Exception):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )

    def test_invalid_test_size(self):
        """Test that test_size outside valid range raises validation error."""
        with pytest.raises(Exception):
            TrainingRequest(target_column="target", test_size=0.6)


class TestModelManager:
    def test_initialization(self, model_manager):
        """Test ModelManager initialization creates directory."""
        assert model_manager.model_dir.exists()
        assert model_manager.model_dir.is_dir()

    def test_train_model_success(self, model_manager, sample_dataframe):
        """Test successful model training with valid data."""
        # Prepare request
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "feature2"],
        )

        # Train model
        response = model_manager.train_model(sample_dataframe, request)

        # Verify response structure
        assert isinstance(response, TrainingResponse)
        assert response.model_id is not None
        assert response.r_squared > 0
        assert response.n_samples == 100
        assert response.n_features == 2
        assert response.feature_columns == ["feature1", "feature2"]
        assert response.target_column == "target"
        assert response.model_path.endswith(".joblib")
        assert Path(response.model_path).exists()

        # Verify metrics
        assert "r2" in response.metrics
        assert "mse" in response.metrics
        assert "rmse" in response.metrics
        assert "mae" in response.metrics

        # Verify coefficients and p-values
        assert len(response.coefficients) == 3  # 2 features + intercept
        assert len(response.p_values) == 3

        # Verify diagnostics
        assert "durbin_watson" in response.diagnostics
        assert "breusch_pagan" in response.diagnostics

    def test_train_model_without_feature_columns(self, model_manager, sample_dataframe):
        """Test training when feature_columns is None (use all except target)."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        response = model_manager.train_model(sample_dataframe, request)

        assert response.n_features == 2
        assert response.feature_columns == ["feature1", "feature2"]

    def test_train_model_missing_target(self, model_manager, sample_dataframe):
        """Test training with non-existent target column raises error."""
        request = TrainingRequest(
            target_column="nonexistent_column",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(HTTPException) as exc_info:
            model_manager.train_model(sample_dataframe, request)

        assert exc_info.value.status_code == 400

    def test_train_model_missing_feature(self, model_manager, sample_dataframe):
        """Test training with non-existent feature column raises error."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["nonexistent_feature"],
        )

        with pytest.raises(HTTPException) as exc_info:
            model_manager.train_model(sample_dataframe, request)

        assert exc_info.value.status_code == 400

    def test_train_model_serialization(self, model_manager, sample_dataframe):
        """Test that trained model is properly serialized and can be loaded."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        response = model_manager.train_model(sample_dataframe, request)

        # Load the serialized model
        loaded_model = joblib.load(response.model_path)
        assert loaded_model is not None

        # Verify model can make predictions
        test_data = sample_dataframe[["feature1", "feature2"]].iloc[:5]
        predictions = loaded_model.predict(test_data)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    def test_train_model_with_mock(self, model_manager, sample_dataframe):
        """Test training with mocked sklearn components."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with patch("src.model_manager.LinearRegression") as mock_lr, \
             patch("src.model_manager.OLS") as mock_ols, \
             patch("src.model_manager.train_test_split") as mock_split:

            # Configure mocks
            mock_lr_instance = MagicMock()
            mock_lr_instance.fit.return_value = None
            mock_lr_instance.predict.return_value = np.array([1.0, 2.0, 3.0])
            mock_lr_instance.coef_ = np.array([0.5, 0.3])
            mock_lr_instance.intercept_ = 0.1
            mock_lr.return_value = mock_lr_instance

            mock_ols_instance = MagicMock()
            mock_ols_instance.fit.return_value = mock_ols_instance
            mock_ols_instance.params = np.array([0.1, 0.5, 0.3])
            mock_ols_instance.pvalues = np.array([0.01, 0.02, 0.03])
            mock_ols_instance.rsquared = 0.85
            mock_ols_instance.rsquared_adj = 0.82
            mock_ols.return_value = mock_ols_instance

            mock_split.return_value = (
                sample_dataframe[["feature1", "feature2"]],
                sample_dataframe[["feature1", "feature2"]],
                sample_dataframe["target"],
                sample_dataframe["target"],
            )

            response = model_manager.train_model(sample_dataframe, request)

            assert response.r_squared == 0.85
            assert response.adjusted_r_squared == 0.82
            assert len(response.coefficients) == 3
            assert len(response.p_values) == 3

    def test_validate_file_size(self, model_manager):
        """Test file size validation."""
        # Create a file larger than MAX_FILE_SIZE
        large_file = MagicMock(spec=UploadFile)
        large_file.size = MAX_FILE_SIZE + 1

        with pytest.raises(HTTPException) as exc_info:
            model_manager.validate_file(large_file)

        assert exc_info.value.status_code == 413

    def test_validate_file_extension(self, model_manager):
        """Test file extension validation."""
        # Create a file with unsupported extension
        invalid_file = MagicMock(spec=UploadFile)
        invalid_file.size = 1000
        invalid_file.filename = "data.json"

        with pytest.raises(HTTPException) as exc_info:
            model_manager.validate_file(invalid_file)

        assert exc_info.value.status_code == 400

    def test_validate_file_success(self, model_manager):
        """Test valid file passes validation."""
        valid_file = MagicMock(spec=UploadFile)
        valid_file.size = 1000
        valid_file.filename = "data.csv"

        # Should not raise any exception
        model_manager.validate_file(valid_file)

    def test_parse_csv(self, model_manager, sample_csv_bytes):
        """Test CSV parsing from bytes."""
        df = model_manager.parse_csv(sample_csv_bytes)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "target" in df.columns
        assert "feature1" in df.columns
        assert "feature2" in df.columns

    def test_parse_csv_invalid_data(self, model_manager):
        """Test CSV parsing with invalid data."""
        invalid_csv = b"not,a,valid,csv\n1,2,3\n4,5"

        with pytest.raises(HTTPException) as exc_info:
            model_manager.parse_csv(invalid_csv)

        assert exc_info.value.status_code == 400