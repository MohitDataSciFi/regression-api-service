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
    ModelManager,
    TrainingRequest,
    TrainingResponse,
    PredictionRequest,
    PredictionResponse,
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
    """Create a ModelManager instance with a temporary model directory."""
    return ModelManager(model_dir=str(tmp_path / "models"))


@pytest.fixture
def trained_model_manager(model_manager, sample_dataframe):
    """Create a ModelManager with a trained model."""
    model_manager.train(
        df=sample_dataframe,
        target_column="target",
        feature_columns=["feature1", "feature2"],
        test_size=0.2,
        random_state=42,
    )
    return model_manager


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
        """Test that an empty target column raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="   ", test_size=0.2)

    def test_invalid_test_size(self):
        """Test that test_size outside [0.1, 0.5] raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="target", test_size=0.6)

    def test_duplicate_feature_columns(self):
        """Test that duplicate feature columns raise validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )


class TestModelManagerTraining:
    def test_train_returns_training_response(self, model_manager, sample_dataframe):
        """Test that training returns a valid TrainingResponse."""
        response = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
        )

        assert isinstance(response, TrainingResponse)
        assert response.model_id
        assert response.training_timestamp
        assert response.r_squared > 0
        assert response.mse >= 0
        assert response.rmse >= 0
        assert response.mae >= 0
        assert "feature1" in response.coefficients
        assert "feature2" in response.coefficients
        assert "feature1" in response.p_values
        assert "feature2" in response.p_values
        assert response.model_path.endswith(".joblib")

    def test_train_serializes_model(self, model_manager, sample_dataframe):
        """Test that training saves the model to disk."""
        response = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
        )

        model_path = Path(response.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Verify the model can be loaded
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

    def test_train_with_default_features(self, model_manager, sample_dataframe):
        """Test training with default feature columns (all except target)."""
        response = model_manager.train(
            df=sample_dataframe,
            target_column="target",
        )

        assert "feature1" in response.coefficients
        assert "feature2" in response.coefficients

    def test_train_invalid_target_column(self, model_manager, sample_dataframe):
        """Test training with non-existent target column raises error."""
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            model_manager.train(
                df=sample_dataframe,
                target_column="nonexistent",
            )

    def test_train_invalid_feature_columns(self, model_manager, sample_dataframe):
        """Test training with non-existent feature column raises error."""
        with pytest.raises(ValueError, match="Feature column 'nonexistent' not found"):
            model_manager.train(
                df=sample_dataframe,
                target_column="target",
                feature_columns=["nonexistent"],
            )


class TestModelManagerPrediction:
    def test_predict_returns_prediction_response(self, trained_model_manager):
        """Test that prediction returns a valid PredictionResponse."""
        features = {"feature1": 1.0, "feature2": 2.0}
        response = trained_model_manager.predict(features)

        assert isinstance(response, PredictionResponse)
        assert isinstance(response.prediction, float)
        assert response.model_id == trained_model_manager.model_id
        assert response.timestamp

    def test_predict_with_missing_features(self, trained_model_manager):
        """Test prediction with missing feature raises error."""
        with pytest.raises(ValueError, match="Missing features"):
            trained_model_manager.predict({"feature1": 1.0})

    def test_predict_with_extra_features(self, trained_model_manager):
        """Test prediction with extra features raises error."""
        with pytest.raises(ValueError, match="Unexpected features"):
            trained_model_manager.predict(
                {"feature1": 1.0, "feature2": 2.0, "extra": 3.0}
            )

    def test_predict_before_training(self, model_manager):
        """Test prediction before training raises error."""
        with pytest.raises(RuntimeError, match="Model not trained"):
            model_manager.predict({"feature1": 1.0, "feature2": 2.0})


class TestModelManagerDiagnostics:
    def test_diagnostics_include_breusch_pagan(self, trained_model_manager):
        """Test that diagnostics include Breusch-Pagan test results."""
        response = trained_model_manager.model_metadata
        assert "breusch_pagan" in response["diagnostics"]
        bp_result = response["diagnostics"]["breusch_pagan"]
        assert "lm_statistic" in bp_result
        assert "lm_pvalue" in bp_result
        assert "f_statistic" in bp_result
        assert "f_pvalue" in bp_result

    def test_diagnostics_include_durbin_watson(self, trained_model_manager):
        """Test that diagnostics include Durbin-Watson statistic."""
        response = trained_model_manager.model_metadata
        assert "durbin_watson" in response["diagnostics"]
        assert isinstance(response["diagnostics"]["durbin_watson"], float)

    def test_feature_importance_matches_coefficients(self, trained_model_manager):
        """Test that feature importance is derived from coefficients."""
        response = trained_model_manager.model_metadata
        for feature in response["coefficients"]:
            assert feature in response["feature_importance"]
            assert response["feature_importance"][feature] == abs(
                response["coefficients"][feature]
            )


class TestModelManagerSerialization:
    def test_save_and_load_model(self, model_manager, sample_dataframe):
        """Test that model can be saved and loaded back."""
        # Train model
        response = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
        )

        # Create a new manager and load the model
        new_manager = ModelManager(model_dir=model_manager.model_dir)
        new_manager.load_model(response.model_id)

        # Verify predictions match
        features = {"feature1": 0.5, "feature2": -0.5}
        pred1 = model_manager.predict(features)
        pred2 = new_manager.predict(features)
        assert pred1.prediction == pytest.approx(pred2.prediction)

    def test_load_nonexistent_model(self, model_manager):
        """Test loading a non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            model_manager.load_model("nonexistent_model_id")


class TestModelManagerUpload:
    @pytest.mark.asyncio
    async def test_train_from_upload(self, model_manager, sample_dataframe):
        """Test training from an uploaded CSV file."""
        # Create a mock UploadFile
        csv_buffer = io.StringIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "data.csv"
        upload_file.read = MagicMock(return_value=csv_bytes)

        response = model_manager.train_from_upload(
            upload_file=upload_file,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
        )

        assert isinstance(response, TrainingResponse)
        assert response.r_squared > 0

    @pytest.mark.asyncio
    async def test_train_from_upload_invalid_csv(self, model_manager):
        """Test training from invalid CSV raises error."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "data.csv"
        upload_file.read = MagicMock(return_value=b"invalid,csv,data\n1,2,3\n")

        with pytest.raises(ValueError, match="Error reading CSV"):
            await model_manager.train_from_upload(
                upload_file=upload_file,
                target_column="target",
            )