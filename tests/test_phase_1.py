import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from src.model_manager import (
    MAX_FILE_SIZE,
    MODEL_DIR,
    SUPPORTED_EXTENSIONS,
    ModelManager,
    TrainingData,
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
    """Create a ModelManager instance with a temporary model directory."""
    return ModelManager(model_dir=tmp_path)


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile with CSV content."""
    csv_content = b"feature1,feature2,target\n1.0,2.0,3.0\n2.0,3.0,5.0\n3.0,4.0,7.0\n4.0,5.0,9.0\n5.0,6.0,11.0\n"
    upload_file = MagicMock(spec=UploadFile)
    upload_file.filename = "test_data.csv"
    upload_file.content_type = "text/csv"
    upload_file.file = io.BytesIO(csv_content)
    return upload_file


class TestTrainingData:
    """Tests for TrainingData Pydantic model."""

    def test_valid_training_data(self):
        """Test valid training data configuration."""
        data = TrainingData(
            target_column="target",
            test_size=0.2,
            random_state=42,
            feature_columns=["feature1", "feature2"],
        )
        assert data.target_column == "target"
        assert data.test_size == 0.2
        assert data.random_state == 42
        assert data.feature_columns == ["feature1", "feature2"]

    def test_invalid_target_column(self):
        """Test validation error for empty target column."""
        with pytest.raises(ValidationError):
            TrainingData(target_column="   ", test_size=0.2)

    def test_invalid_test_size(self):
        """Test validation error for test_size out of range."""
        with pytest.raises(ValidationError):
            TrainingData(target_column="target", test_size=0.6)

    def test_duplicate_feature_columns(self):
        """Test validation error for duplicate feature columns."""
        with pytest.raises(ValidationError):
            TrainingData(
                target_column="target",
                feature_columns=["feature1", "feature1"],
            )


class TestModelManager:
    """Tests for ModelManager class."""

    def test_train_model_success(self, model_manager, sample_dataframe):
        """Test successful model training with valid data."""
        # Train model
        metrics = model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        # Verify metrics
        assert metrics.r2_score > 0.5
        assert metrics.mse > 0
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert metrics.training_samples == 80
        assert metrics.test_samples == 20
        assert metrics.feature_count == 2
        assert metrics.model_version == "1.0.0"
        assert metrics.created_at is not None
        assert "feature1" in metrics.coefficients
        assert "feature2" in metrics.coefficients
        assert "feature1" in metrics.p_values
        assert "feature2" in metrics.p_values

    def test_train_model_with_feature_selection(self, model_manager, sample_dataframe):
        """Test model training with specific feature columns."""
        metrics = model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            feature_columns=["feature1"],
            test_size=0.2,
            random_state=42,
        )

        assert metrics.feature_count == 1
        assert "feature1" in metrics.coefficients
        assert "feature2" not in metrics.coefficients

    def test_train_model_invalid_target(self, model_manager, sample_dataframe):
        """Test model training with non-existent target column."""
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            model_manager.train_model(
                df=sample_dataframe,
                target_column="nonexistent",
                test_size=0.2,
                random_state=42,
            )

    def test_train_model_invalid_feature(self, model_manager, sample_dataframe):
        """Test model training with non-existent feature column."""
        with pytest.raises(ValueError, match="Feature column 'nonexistent' not found"):
            model_manager.train_model(
                df=sample_dataframe,
                target_column="target",
                feature_columns=["nonexistent"],
                test_size=0.2,
                random_state=42,
            )

    def test_save_and_load_model(self, model_manager, sample_dataframe):
        """Test model serialization and deserialization."""
        # Train model
        metrics = model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        # Save model
        model_path = model_manager.save_model(
            model=model_manager.model,
            metrics=metrics,
            filename="test_model.joblib",
        )
        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        # Load model
        loaded_model = model_manager.load_model(model_path)
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")

        # Verify predictions match
        X_test = sample_dataframe[["feature1", "feature2"]].iloc[:5]
        original_preds = model_manager.model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_model_invalid_extension(self, model_manager, sample_dataframe):
        """Test saving model with unsupported extension."""
        metrics = model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValueError, match="Unsupported model file extension"):
            model_manager.save_model(
                model=model_manager.model,
                metrics=metrics,
                filename="test_model.pkl",
            )

    def test_load_model_not_found(self, model_manager):
        """Test loading non-existent model file."""
        with pytest.raises(FileNotFoundError):
            model_manager.load_model(Path("nonexistent_model.joblib"))

    def test_validate_upload_file_success(self, model_manager, mock_upload_file):
        """Test successful file validation."""
        df = model_manager.validate_upload_file(mock_upload_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["feature1", "feature2", "target"]

    def test_validate_upload_file_invalid_extension(self, model_manager):
        """Test file validation with unsupported extension."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "test_data.xlsx"
        upload_file.content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        upload_file.file = io.BytesIO(b"some content")

        with pytest.raises(HTTPException, match="Unsupported file type"):
            model_manager.validate_upload_file(upload_file)

    def test_validate_upload_file_empty(self, model_manager):
        """Test file validation with empty file."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "empty.csv"
        upload_file.content_type = "text/csv"
        upload_file.file = io.BytesIO(b"")

        with pytest.raises(HTTPException, match="File is empty"):
            model_manager.validate_upload_file(upload_file)

    def test_validate_upload_file_too_large(self, model_manager):
        """Test file validation with oversized file."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "large.csv"
        upload_file.content_type = "text/csv"
        upload_file.file = io.BytesIO(b"x" * (MAX_FILE_SIZE + 1))

        with pytest.raises(HTTPException, match="File size exceeds"):
            model_manager.validate_upload_file(upload_file)

    def test_validate_upload_file_invalid_csv(self, model_manager):
        """Test file validation with malformed CSV."""
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "malformed.csv"
        upload_file.content_type = "text/csv"
        upload_file.file = io.BytesIO(b"col1,col2\n1,2\n3")

        with pytest.raises(HTTPException, match="Invalid CSV"):
            model_manager.validate_upload_file(upload_file)

    def test_get_model_metrics(self, model_manager, sample_dataframe):
        """Test retrieving model metrics after training."""
        # Train model first
        model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        metrics = model_manager.get_model_metrics()
        assert metrics is not None
        assert metrics.r2_score > 0
        assert metrics.training_samples == 80
        assert metrics.test_samples == 20

    def test_get_model_metrics_no_model(self, model_manager):
        """Test retrieving metrics when no model has been trained."""
        with pytest.raises(ValueError, match="No model has been trained"):
            model_manager.get_model_metrics()

    def test_get_model_path(self, model_manager):
        """Test getting the model path."""
        path = model_manager.get_model_path("test_model.joblib")
        assert path == model_manager.model_dir / "test_model.joblib"

    def test_list_models(self, model_manager, sample_dataframe):
        """Test listing saved models."""
        # Train and save a model
        metrics = model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        model_manager.save_model(
            model=model_manager.model,
            metrics=metrics,
            filename="test_model.joblib",
        )

        models = model_manager.list_models()
        assert len(models) == 1
        assert models[0]["filename"] == "test_model.joblib"
        assert models[0]["size"] > 0
        assert models[0]["created_at"] is not None

    def test_list_models_empty(self, model_manager):
        """Test listing models when none exist."""
        models = model_manager.list_models()
        assert models == []

    def test_delete_model(self, model_manager, sample_dataframe):
        """Test deleting a saved model."""
        # Train and save a model
        metrics = model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        model_path = model_manager.save_model(
            model=model_manager.model,
            metrics=metrics,
            filename="test_model.joblib",
        )

        # Delete the model
        result = model_manager.delete_model("test_model.joblib")
        assert result is True
        assert not model_path.exists()

    def test_delete_model_not_found(self, model_manager):
        """Test deleting a non-existent model."""
        with pytest.raises(FileNotFoundError):
            model_manager.delete_model("nonexistent.joblib")

    def test_get_model_info(self, model_manager, sample_dataframe):
        """Test getting model information."""
        # Train a model
        metrics = model_manager.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        info = model_manager.get_model_info()
        assert info["model_version"] == "1.0.0"
        assert info["feature_count"] == 2
        assert info["training_samples"] == 80
        assert info["test_samples"] == 20
        assert info["r2_score"] > 0
        assert info["created_at"] is not None

    def test_get_model_info_no_model(self, model_manager):
        """Test getting model info when no model exists."""
        with pytest.raises(ValueError, match="No model has been trained"):
            model_manager.get_model_info()

    def test_train_model_with_logging(self, model_manager, sample_dataframe, caplog):
        """Test that training logs appropriate messages."""
        with caplog.at_level(logging.INFO):
            model_manager.train_model(
                df=sample_dataframe,
                target_column="target",
                test_size=0.2,
                random_state=42,
            )

        assert "Training model" in caplog.text
        assert "Model training completed" in caplog.text
        assert "R² score" in caplog.text

    def test_train_model_error_handling(self, model_manager, sample_dataframe):
        """Test error handling during training with invalid data."""
        # Create dataframe with NaN values
        df_with_nan = sample_dataframe.copy()
        df_with_nan.loc[0, "feature1"] = np.nan

        with pytest.raises(ValueError, match="Data contains NaN values"):
            model_manager.train_model(
                df=df_with_nan,
                target_column="target",
                test_size=0.2,
                random_state=42,
            )

    def test_train_model_with_categorical_features(self, model_manager):
        """Test training with categorical features."""
        data = pd.DataFrame({
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        })

        with pytest.raises(ValueError, match="Categorical columns"):
            model_manager.train_model(
                df=data,
                target_column="target",
                test_size=0.2,
                random_state=42,
            )

    def test_model_persistence_across_instances(self, tmp_path, sample_dataframe):
        """Test that models persist across ModelManager instances."""
        # First instance - train and save
        manager1 = ModelManager(model_dir=tmp_path)
        metrics = manager1.train_model(
            df=sample_dataframe,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )
        model_path = manager1.save_model(
            model=manager1.model,
            metrics=metrics,
            filename="persistent_model.joblib",
        )

        # Second instance - load and predict
        manager2 = ModelManager(model_dir=tmp_path)
        loaded_model = manager2.load_model(model_path)

        # Verify predictions match
        X_test = sample_dataframe[["feature1", "feature2"]].iloc[:5]
        original_preds = manager1.model.predict(X_test)
        loaded_preds = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)