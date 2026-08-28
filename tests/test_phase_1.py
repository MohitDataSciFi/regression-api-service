import io
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from src.model_manager import (
    ModelManager,
    ModelTrainingError,
    ModelValidationError,
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
def trained_model_manager(model_manager, sample_dataframe):
    """Create a ModelManager with a trained model."""
    model_id = model_manager.train(
        df=sample_dataframe,
        target_column="target",
        feature_columns=["feature1", "feature2"],
        test_size=0.2,
        random_state=42,
    )
    return model_manager, model_id


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

    def test_invalid_target_column(self):
        """Test that empty target column raises validation error."""
        with pytest.raises(ValidationError):
            TrainingRequest(target_column="   ")

    def test_invalid_test_size(self):
        """Test that test_size outside bounds raises validation error."""
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
    """Tests for ModelManager training functionality."""

    def test_train_success(self, model_manager, sample_dataframe):
        """Test successful model training."""
        model_id = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
        )

        # Verify model was saved
        model_path = model_manager.model_dir / f"{model_id}.joblib"
        assert model_path.exists()

        # Verify model can be loaded
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict")

        # Verify training metadata was saved
        metadata_path = model_manager.model_dir / f"{model_id}_metadata.json"
        assert metadata_path.exists()

    def test_train_without_feature_columns(self, model_manager, sample_dataframe):
        """Test training without specifying feature columns (uses all except target)."""
        model_id = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        model_path = model_manager.model_dir / f"{model_id}.joblib"
        assert model_path.exists()

    def test_train_invalid_target(self, model_manager, sample_dataframe):
        """Test training with non-existent target column."""
        with pytest.raises(ModelValidationError):
            model_manager.train(
                df=sample_dataframe,
                target_column="nonexistent_column",
                test_size=0.2,
                random_state=42,
            )

    def test_train_invalid_feature_columns(self, model_manager, sample_dataframe):
        """Test training with non-existent feature columns."""
        with pytest.raises(ModelValidationError):
            model_manager.train(
                df=sample_dataframe,
                target_column="target",
                feature_columns=["nonexistent_feature"],
                test_size=0.2,
                random_state=42,
            )

    def test_train_returns_valid_response(self, model_manager, sample_dataframe):
        """Test that training returns a valid TrainingResponse."""
        response = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
        )

        assert isinstance(response, TrainingResponse)
        assert response.r_squared > 0
        assert response.mse >= 0
        assert response.rmse >= 0
        assert response.mae >= 0
        assert response.sample_size == len(sample_dataframe)
        assert response.feature_count == 2
        assert "feature1" in response.coefficients
        assert "feature2" in response.coefficients
        assert "feature1" in response.p_values
        assert "feature2" in response.p_values
        assert response.model_path.endswith(".joblib")
        assert "breusch_pagan" in response.diagnostics
        assert "durbin_watson" in response.diagnostics


class TestModelManagerPrediction:
    """Tests for ModelManager prediction functionality."""

    def test_predict_success(self, trained_model_manager, sample_dataframe):
        """Test successful prediction with trained model."""
        model_manager, model_id = trained_model_manager

        # Create test data
        test_data = pd.DataFrame({
            "feature1": [0.5, -0.3],
            "feature2": [1.2, -0.8],
        })

        predictions = model_manager.predict(model_id, test_data)

        assert len(predictions) == 2
        assert all(isinstance(p, (int, float)) for p in predictions)

    def test_predict_nonexistent_model(self, model_manager, sample_dataframe):
        """Test prediction with non-existent model ID."""
        with pytest.raises(ModelTrainingError):
            model_manager.predict("nonexistent_model", sample_dataframe)

    def test_predict_missing_features(self, trained_model_manager):
        """Test prediction with missing feature columns."""
        model_manager, model_id = trained_model_manager

        test_data = pd.DataFrame({
            "feature1": [0.5, -0.3],
            # Missing feature2
        })

        with pytest.raises(ModelValidationError):
            model_manager.predict(model_id, test_data)


class TestModelManagerSerialization:
    """Tests for model serialization and loading."""

    def test_save_and_load_model(self, model_manager, sample_dataframe):
        """Test that model can be saved and loaded correctly."""
        model_id = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
        )

        # Load the model
        loaded_model = model_manager.load_model(model_id)

        # Verify it's a sklearn model
        from sklearn.linear_model import LinearRegression
        assert isinstance(loaded_model, LinearRegression)

        # Test prediction consistency
        test_data = pd.DataFrame({
            "feature1": [0.1, 0.2],
            "feature2": [0.3, 0.4],
        })

        predictions_before = model_manager.predict(model_id, test_data)
        predictions_after = loaded_model.predict(test_data[["feature1", "feature2"]])

        np.testing.assert_array_almost_equal(predictions_before, predictions_after)

    def test_model_metadata_saved(self, model_manager, sample_dataframe):
        """Test that model metadata is saved correctly."""
        model_id = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
        )

        metadata_path = model_manager.model_dir / f"{model_id}_metadata.json"
        assert metadata_path.exists()

        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["model_id"] == model_id
        assert metadata["target_column"] == "target"
        assert metadata["feature_columns"] == ["feature1", "feature2"]
        assert metadata["test_size"] == 0.2
        assert metadata["random_state"] == 42
        assert "training_timestamp" in metadata
        assert "metrics" in metadata


class TestModelManagerDiagnostics:
    """Tests for model diagnostics."""

    def test_diagnostics_are_computed(self, model_manager, sample_dataframe):
        """Test that diagnostics are computed and returned."""
        response = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1", "feature2"],
            test_size=0.2,
            random_state=42,
        )

        # Check R-squared
        assert 0 <= response.r_squared <= 1

        # Check adjusted R-squared
        assert 0 <= response.adjusted_r_squared <= 1

        # Check coefficients
        assert len(response.coefficients) == 2
        assert all(isinstance(v, float) for v in response.coefficients.values())

        # Check p-values
        assert len(response.p_values) == 2
        assert all(isinstance(v, float) for v in response.p_values.values())

        # Check diagnostics
        assert "breusch_pagan" in response.diagnostics
        assert "durbin_watson" in response.diagnostics
        assert isinstance(response.diagnostics["breusch_pagan"], dict)
        assert isinstance(response.diagnostics["durbin_watson"], float)

    def test_diagnostics_with_single_feature(self, model_manager, sample_dataframe):
        """Test diagnostics with a single feature."""
        response = model_manager.train(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1"],
            test_size=0.2,
            random_state=42,
        )

        assert response.feature_count == 1
        assert len(response.coefficients) == 1
        assert len(response.p_values) == 1


class TestModelManagerErrorHandling:
    """Tests for error handling in ModelManager."""

    def test_train_with_empty_dataframe(self, model_manager):
        """Test training with empty dataframe."""
        empty_df = pd.DataFrame()
        with pytest.raises(ModelValidationError):
            model_manager.train(
                df=empty_df,
                target_column="target",
                test_size=0.2,
                random_state=42,
            )

    def test_train_with_insufficient_data(self, model_manager):
        """Test training with too few samples."""
        small_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "target": [1, 2, 3],
        })
        with pytest.raises(ModelValidationError):
            model_manager.train(
                df=small_df,
                target_column="target",
                test_size=0.2,
                random_state=42,
            )

    def test_train_with_constant_target(self, model_manager):
        """Test training with constant target variable."""
        constant_df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [1, 1, 1, 1, 1],
        })
        with pytest.raises(ModelTrainingError):
            model_manager.train(
                df=constant_df,
                target_column="target",
                test_size=0.2,
                random_state=42,
            )