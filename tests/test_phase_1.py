import io
import joblib
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.model_manager import ModelManager, ModelArtifacts, TrainingRequest


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randn(n_samples) * 2 + 1
    })
    return data


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with temporary directory."""
    return ModelManager(model_dir=str(tmp_path / "models"))


class TestModelManager:
    def test_train_model_returns_valid_artifacts(self, model_manager, sample_data):
        """Test that training produces valid model artifacts with expected structure."""
        # Arrange
        request = TrainingRequest(
            target_column='target',
            test_size=0.2,
            random_state=42,
            feature_columns=['feature1', 'feature2', 'feature3']
        )

        # Act
        artifacts = model_manager.train(sample_data, request)

        # Assert
        assert isinstance(artifacts, ModelArtifacts)
        assert artifacts.model_id is not None
        assert artifacts.sklearn_model is not None
        assert artifacts.statsmodels_model is not None
        assert artifacts.feature_names == ['feature1', 'feature2', 'feature3']
        assert artifacts.target_name == 'target'
        assert artifacts.model_path.endswith('.joblib')
        assert Path(artifacts.model_path).exists()
        
        # Check metrics
        assert 'r2' in artifacts.metrics
        assert 'mse' in artifacts.metrics
        assert 'mae' in artifacts.metrics
        assert 'rmse' in artifacts.metrics
        assert 'adjusted_r2' in artifacts.metrics
        assert 'durbin_watson' in artifacts.metrics
        assert 'breusch_pagan_pvalue' in artifacts.metrics
        
        # Check coefficients and p-values
        assert len(artifacts.coefficients) == 3  # 3 features
        assert len(artifacts.p_values) == 3
        assert all(feature in artifacts.coefficients for feature in ['feature1', 'feature2', 'feature3'])
        assert all(feature in artifacts.p_values for feature in ['feature1', 'feature2', 'feature3'])

    def test_train_model_with_default_features(self, model_manager, sample_data):
        """Test that training works when feature_columns is None (uses all except target)."""
        # Arrange
        request = TrainingRequest(
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        # Act
        artifacts = model_manager.train(sample_data, request)

        # Assert
        assert artifacts.feature_names == ['feature1', 'feature2', 'feature3']
        assert len(artifacts.coefficients) == 3

    def test_model_serialization_and_loading(self, model_manager, sample_data):
        """Test that model can be serialized and loaded back correctly."""
        # Arrange
        request = TrainingRequest(
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        artifacts = model_manager.train(sample_data, request)

        # Act
        loaded_model = joblib.load(artifacts.model_path)

        # Assert
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        # Test prediction consistency
        test_data = sample_data[artifacts.feature_names].iloc[:5]
        original_pred = artifacts.sklearn_model.predict(test_data)
        loaded_pred = loaded_model.predict(test_data)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_train_model_with_invalid_target(self, model_manager, sample_data):
        """Test that training raises error for invalid target column."""
        # Arrange
        request = TrainingRequest(
            target_column='nonexistent_column',
            test_size=0.2,
            random_state=42
        )

        # Act & Assert
        with pytest.raises(KeyError):
            model_manager.train(sample_data, request)

    def test_train_model_with_insufficient_data(self, model_manager):
        """Test that training handles insufficient data gracefully."""
        # Arrange
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [1, 2, 3]
        })
        request = TrainingRequest(
            target_column='target',
            test_size=0.5,
            random_state=42
        )

        # Act & Assert
        with pytest.raises(ValueError):
            model_manager.train(small_data, request)

    def test_model_metrics_quality(self, model_manager, sample_data):
        """Test that model metrics are reasonable for a linear relationship."""
        # Arrange - create data with strong linear relationship
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 2)
        y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + np.random.randn(n_samples) * 0.1
        
        data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })
        
        request = TrainingRequest(
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        # Act
        artifacts = model_manager.train(data, request)

        # Assert
        assert artifacts.metrics['r2'] > 0.9  # High R² for strong linear relationship
        assert artifacts.metrics['mse'] < 0.1  # Low MSE
        assert artifacts.metrics['mae'] < 0.2  # Low MAE
        assert artifacts.metrics['rmse'] < 0.3  # Low RMSE
        
        # Check that feature1 has higher importance than feature2
        assert abs(artifacts.coefficients['feature1']) > abs(artifacts.coefficients['feature2'])