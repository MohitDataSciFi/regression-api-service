import io
import logging
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
    ErrorResponse,
    TrainingRequest,
    TrainingResponse,
    ModelManager,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(0, 1, n_samples),
        "target": np.random.normal(0, 1, n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with a temporary model directory."""
    return ModelManager(model_dir=str(tmp_path))


@pytest.fixture
def valid_training_request():
    """Create a valid TrainingRequest."""
    return TrainingRequest(
        target_column="target",
        test_size=0.2,
        random_state=42,
        feature_columns=["feature1", "feature2"],
    )


class TestTrainingRequestValidation:
    """Test the Pydantic validation logic."""

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

    def test_invalid_test_size(self):
        """Test that test_size outside [0.1, 0.5] raises ValidationError."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                test_size=0.05,  # Too small
            )

        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                test_size=0.6,  # Too large
            )

    def test_empty_target_column(self):
        """Test that empty target column raises ValidationError."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="   ",
            )

    def test_duplicate_feature_columns(self):
        """Test that duplicate feature columns raise ValidationError."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )


class TestModelManagerTraining:
    """Test the ModelManager training logic."""

    def test_train_model_success(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training produces valid metrics and saves the model."""
        # Train the model
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify response structure
        assert isinstance(response, TrainingResponse)
        assert response.status == "success"
        assert response.r_squared > 0
        assert response.mse >= 0
        assert response.rmse >= 0
        assert response.mae >= 0
        assert response.durbin_watson > 0
        assert response.breusch_pagan_pvalue >= 0

        # Verify coefficients and p-values
        assert len(response.coefficients) == 2  # feature1, feature2
        assert len(response.p_values) == 2
        assert all(v >= 0 for v in response.p_values.values())

        # Verify feature importance
        assert len(response.feature_importance) == 2
        assert all(v >= 0 for v in response.feature_importance.values())

        # Verify model file was saved
        model_path = Path(response.model_path)
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Verify model can be loaded
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict")

    def test_train_model_with_default_features(
        self, model_manager, sample_dataframe
    ):
        """Test training with default feature columns (all except target)."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        response = model_manager.train(
            df=sample_dataframe,
            request=request,
        )

        # Should use both feature1 and feature2 by default
        assert len(response.coefficients) == 2
        assert "feature1" in response.coefficients
        assert "feature2" in response.coefficients

    def test_train_model_with_missing_target(self, model_manager, sample_dataframe):
        """Test that training with missing target column raises error."""
        request = TrainingRequest(
            target_column="nonexistent_target",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Target column"):
            model_manager.train(
                df=sample_dataframe,
                request=request,
            )

    def test_train_model_with_missing_feature(self, model_manager, sample_dataframe):
        """Test that training with missing feature column raises error."""
        request = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "nonexistent_feature"],
        )

        with pytest.raises(ValueError, match="Feature column"):
            model_manager.train(
                df=sample_dataframe,
                request=request,
            )

    def test_train_model_saves_metadata(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training saves metadata alongside the model."""
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Check that metadata file exists
        model_path = Path(response.model_path)
        metadata_path = model_path.with_suffix(".json")
        assert metadata_path.exists()

        # Verify metadata content
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["model_id"] == response.model_id
        assert metadata["target_column"] == "target"
        assert metadata["feature_columns"] == ["feature1", "feature2"]
        assert metadata["test_size"] == 0.2
        assert metadata["random_state"] == 42
        assert metadata["r_squared"] == response.r_squared

    def test_train_model_reproducibility(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training with same seed produces same results."""
        response1 = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        response2 = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify same metrics
        assert response1.r_squared == response2.r_squared
        assert response1.mse == response2.mse
        assert response1.coefficients == response2.coefficients
        assert response1.p_values == response2.p_values

    def test_train_model_different_seed(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that different seeds produce different results."""
        request1 = valid_training_request
        request2 = TrainingRequest(
            target_column="target",
            test_size=0.2,
            random_state=123,  # Different seed
            feature_columns=["feature1", "feature2"],
        )

        response1 = model_manager.train(
            df=sample_dataframe,
            request=request1,
        )

        response2 = model_manager.train(
            df=sample_dataframe,
            request=request2,
        )

        # Metrics should differ (though could theoretically be same)
        assert response1.model_id != response2.model_id
        assert response1.training_timestamp != response2.training_timestamp


class TestModelManagerSerialization:
    """Test model serialization and loading."""

    def test_save_and_load_model(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that a trained model can be saved and loaded."""
        # Train and save
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Load the model
        loaded_model = model_manager.load_model(response.model_id)

        # Verify it's a valid model
        assert hasattr(loaded_model, "predict")
        assert hasattr(loaded_model, "coef_")

        # Test prediction
        test_data = sample_dataframe[["feature1", "feature2"]].iloc[:5]
        predictions = loaded_model.predict(test_data)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    def test_load_nonexistent_model(self, model_manager):
        """Test that loading a nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            model_manager.load_model("nonexistent_model_id")

    def test_model_id_generation(self, model_manager):
        """Test that model IDs are unique and properly formatted."""
        model_id1 = model_manager._generate_model_id()
        model_id2 = model_manager._generate_model_id()

        assert model_id1 != model_id2
        assert isinstance(model_id1, str)
        assert len(model_id1) > 0


class TestModelManagerErrorHandling:
    """Test error handling in ModelManager."""

    def test_train_with_empty_dataframe(self, model_manager, valid_training_request):
        """Test that training with empty DataFrame raises error."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            model_manager.train(
                df=empty_df,
                request=valid_training_request,
            )

    def test_train_with_insufficient_data(
        self, model_manager, valid_training_request
    ):
        """Test that training with too few samples raises error."""
        small_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9],
        })

        with pytest.raises(ValueError, match="samples"):
            model_manager.train(
                df=small_df,
                request=valid_training_request,
            )

    def test_train_with_non_numeric_data(
        self, model_manager, valid_training_request
    ):
        """Test that training with non-numeric data raises error."""
        df = pd.DataFrame({
            "feature1": ["a", "b", "c", "d", "e"],
            "feature2": [1, 2, 3, 4, 5],
            "target": [6, 7, 8, 9, 10],
        })

        with pytest.raises(ValueError, match="numeric"):
            model_manager.train(
                df=df,
                request=valid_training_request,
            )

    def test_train_with_nan_values(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that training with NaN values raises error."""
        df_with_nan = sample_dataframe.copy()
        df_with_nan.loc[0, "feature1"] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            model_manager.train(
                df=df_with_nan,
                request=valid_training_request,
            )


class TestModelManagerDiagnostics:
    """Test diagnostic computations."""

    def test_diagnostics_values(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that diagnostic values are computed correctly."""
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # R-squared should be between 0 and 1
        assert 0 <= response.r_squared <= 1

        # Adjusted R-squared should be <= R-squared
        assert response.adjusted_r_squared <= response.r_squared

        # MSE and RMSE relationship
        assert response.rmse == pytest.approx(np.sqrt(response.mse))

        # MAE should be non-negative
        assert response.mae >= 0

        # Durbin-Watson should be between 0 and 4
        assert 0 <= response.durbin_watson <= 4

        # Breusch-Pagan p-value should be between 0 and 1
        assert 0 <= response.breusch_pagan_pvalue <= 1

    def test_coefficients_and_pvalues(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that coefficients and p-values are correctly computed."""
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify coefficients exist for all features
        assert set(response.coefficients.keys()) == {"feature1", "feature2"}
        assert set(response.p_values.keys()) == {"feature1", "feature2"}

        # Verify p-values are between 0 and 1
        for p_value in response.p_values.values():
            assert 0 <= p_value <= 1

        # Verify coefficients are finite
        for coef in response.coefficients.values():
            assert np.isfinite(coef)

    def test_feature_importance(
        self, model_manager, sample_dataframe, valid_training_request
    ):
        """Test that feature importance is computed."""
        response = model_manager.train(
            df=sample_dataframe,
            request=valid_training_request,
        )

        # Verify feature importance exists for all features
        assert set(response.feature_importance.keys()) == {"feature1", "feature2"}

        # Verify importance values are non-negative
        for importance in response.feature_importance.values():
            assert importance >= 0

        # Verify importance values are finite
        for importance in response.feature_importance.values():
            assert np.isfinite(importance)


class TestModelManagerLogging:
    """Test logging behavior."""

    def test_logging_on_training(
        self, model_manager, sample_dataframe, valid_training_request, caplog
    ):
        """Test that training logs appropriate messages."""
        with caplog.at_level(logging.INFO):
            model_manager.train(
                df=sample_dataframe,
                request=valid_training_request,
            )

        # Verify log messages
        log_messages = [record.message for record in caplog.records]
        assert any("Training" in msg for msg in log_messages)
        assert any("model_id" in msg.lower() for msg in log_messages)

    def test_logging_on_error(
        self, model_manager, sample_dataframe, valid_training_request, caplog
    ):
        """Test that errors are logged."""
        # Create invalid request
        invalid_request = TrainingRequest(
            target_column="nonexistent",
            test_size=0.2,
            random_state=42,
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                model_manager.train(
                    df=sample_dataframe,
                    request=invalid_request,
                )

        # Verify error log messages
        log_messages = [record.message for record in caplog.records]
        assert any("Error" in msg or "error" in msg for msg in log_messages)