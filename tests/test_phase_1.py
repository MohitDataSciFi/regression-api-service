import asyncio
import io
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.model_manager import ModelManager, TrainingRequest, TrainingResponse


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'target': np.random.randn(n_samples) * 2 + 1
    }
    return pd.DataFrame(data)


@pytest.fixture
def model_manager(tmp_path):
    """Create a ModelManager instance with a temporary directory."""
    return ModelManager(model_dir=Path(tmp_path))


@pytest.fixture
def valid_training_request():
    """Create a valid TrainingRequest instance."""
    return TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=['feature1', 'feature2']
    )


@pytest.mark.asyncio
async def test_train_model_success(model_manager, sample_dataframe, valid_training_request):
    """Test successful model training with valid inputs."""
    # Convert DataFrame to CSV bytes for upload simulation
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    # Mock the file upload
    mock_file = MagicMock()
    mock_file.filename = "test_data.csv"
    mock_file.content_type = "text/csv"
    mock_file.file = io.BytesIO(csv_bytes)

    # Train the model
    response = await model_manager.train_model(
        file=mock_file,
        request=valid_training_request
    )

    # Assert response structure
    assert isinstance(response, TrainingResponse)
    assert response.model_id is not None
    assert response.r_squared > 0
    assert response.n_samples == 100
    assert response.n_features == 2
    assert response.feature_columns == ['feature1', 'feature2']
    assert response.target_column == 'target'
    assert response.model_path.endswith('.joblib')

    # Verify model was saved
    model_path = Path(response.model_path)
    assert model_path.exists()
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, 'predict')


@pytest.mark.asyncio
async def test_train_model_invalid_target_column(model_manager, sample_dataframe):
    """Test training with invalid target column."""
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    mock_file = MagicMock()
    mock_file.filename = "test_data.csv"
    mock_file.content_type = "text/csv"
    mock_file.file = io.BytesIO(csv_bytes)

    # Create request with non-existent target column
    request = TrainingRequest(
        target_column='non_existent_column',
        test_size=0.2,
        random_state=42
    )

    with pytest.raises(Exception) as exc_info:
        await model_manager.train_model(file=mock_file, request=request)

    assert "not found" in str(exc_info.value).lower() or "column" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_train_model_without_feature_columns(model_manager, sample_dataframe):
    """Test training when feature_columns is None (should use all except target)."""
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    mock_file = MagicMock()
    mock_file.filename = "test_data.csv"
    mock_file.content_type = "text/csv"
    mock_file.file = io.BytesIO(csv_bytes)

    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42,
        feature_columns=None
    )

    response = await model_manager.train_model(file=mock_file, request=request)

    # Should use all columns except target
    assert response.feature_columns == ['feature1', 'feature2']
    assert response.n_features == 2
    assert response.r_squared > 0


@pytest.mark.asyncio
async def test_model_serialization_and_loading(model_manager, sample_dataframe, valid_training_request):
    """Test that trained model can be serialized and loaded correctly."""
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    mock_file = MagicMock()
    mock_file.filename = "test_data.csv"
    mock_file.content_type = "text/csv"
    mock_file.file = io.BytesIO(csv_bytes)

    response = await model_manager.train_model(
        file=mock_file,
        request=valid_training_request
    )

    # Load the saved model
    loaded_model = joblib.load(response.model_path)

    # Test prediction with the loaded model
    test_data = pd.DataFrame({
        'feature1': [0.5, -0.3],
        'feature2': [1.0, -0.7]
    })

    predictions = loaded_model.predict(test_data)
    assert len(predictions) == 2
    assert all(np.isfinite(predictions))

    # Verify model metadata is stored
    assert response.model_id in model_manager._models
    stored_model = model_manager._models[response.model_id]
    assert stored_model['model_path'] == response.model_path
    assert stored_model['metrics']['r_squared'] == response.r_squared


@pytest.mark.asyncio
async def test_train_model_with_invalid_file_type(model_manager, sample_dataframe):
    """Test training with unsupported file type."""
    csv_buffer = io.StringIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode()

    mock_file = MagicMock()
    mock_file.filename = "test_data.pdf"  # Unsupported extension
    mock_file.content_type = "application/pdf"
    mock_file.file = io.BytesIO(csv_bytes)

    request = TrainingRequest(
        target_column='target',
        test_size=0.2,
        random_state=42
    )

    with pytest.raises(Exception) as exc_info:
        await model_manager.train_model(file=mock_file, request=request)

    assert "unsupported" in str(exc_info.value).lower() or "extension" in str(exc_info.value).lower()