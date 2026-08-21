import asyncio
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError, validator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("regression_api.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Schemas (Pydantic models)
# =============================================================================
class TrainingRequest(BaseModel):
    """Validated training request parameters."""
    target_column: str = Field(..., min_length=1, description="Name of target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test split ratio")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(
        None, description="List of feature columns (default: all except target)"
    )

    @validator("target_column")
    def validate_target_column(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()

    @validator("feature_columns")
    def validate_feature_columns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if len(v) == 0:
                raise ValueError("Feature columns list cannot be empty")
            if len(set(v)) != len(v):
                raise ValueError("Feature columns must be unique")
        return v


class TrainingMetrics(BaseModel):
    """Training metrics returned by the API."""
    r2_score: float
    adjusted_r2: float
    mse: float
    rmse: float
    mae: float
    coefficients: Dict[str, float]
    intercept: float
    p_values: Dict[str, float]
    training_samples: int
    test_samples: int
    feature_count: int
    training_duration_seconds: float
    model_version: str
    timestamp: str


class ModelInfo(BaseModel):
    """Model metadata for serialization."""
    model_id: str
    created_at: str
    metrics: TrainingMetrics
    feature_names: List[str]
    target_name: str
    model_type: str = "linear_regression"


# =============================================================================
# Model Manager
# =============================================================================
class RegressionModelManager:
    """Manages training, diagnostics, and serialization of regression models."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model_cache: Dict[str, Any] = {}
        logger.info(f"Model manager initialized with directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[Any, TrainingMetrics, ModelInfo]:
        """
        Train an OLS regression model with full diagnostics.

        Args:
            data: Input DataFrame
            target_column: Name of target variable
            test_size: Test split ratio
            random_state: Random seed
            feature_columns: Optional list of feature columns

        Returns:
            Tuple of (model, metrics, model_info)
        """
        start_time = time.time()
        logger.info(f"Starting model training with target: {target_column}")

        # Validate data
        if data.empty:
            raise ValueError("Input data is empty")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Prepare features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        else:
            # Validate feature columns exist
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

        # Extract features and target
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Handle missing values
        if X.isnull().any().any() or y.isnull().any():
            logger.warning("Missing values detected, dropping rows")
            mask = X.notnull().all(axis=1) & y.notnull()
            X = X[mask]
            y = y[mask]

        # Validate numeric data
        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")

        # Drop any remaining NaN after conversion
        mask = X.notnull().all(axis=1) & y.notnull()
        X = X[mask]
        y = y[mask]

        if len(X) < 10:
            raise ValueError("Insufficient data after cleaning (minimum 10 samples required)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train scikit-learn model
        sklearn_model = LinearRegression()
        sklearn_model.fit(X_train, y_train)

        # Train statsmodels for diagnostics
        X_train_sm = sm.add_constant(X_train)
        sm_model = sm.OLS(y_train, X_train_sm).fit()

        # Compute predictions and metrics
        y_pred = sklearn_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # Adjusted R²
        n = len(X_test)
        p = X_test.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Extract coefficients and p-values
        coefficients = dict(zip(feature_columns, sklearn_model.coef_))
        intercept = float(sklearn_model.intercept_)

        # Get p-values from statsmodels
        p_values = {}
        for i, col in enumerate(feature_columns):
            p_values[col] = float(sm_model.pvalues[i + 1])  # +1 for constant term

        # Create metrics object
        metrics = TrainingMetrics(
            r2_score=float(r2),
            adjusted_r2=float(adjusted_r2),
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            coefficients=coefficients,
            intercept=intercept,
            p_values=p_values,
            training_samples=len(X_train),
            test_samples=len(X_test),
            feature_count=len(feature_columns),
            training_duration_seconds=time.time() - start_time,
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat(),
        )

        # Create model info
        model_id = f"reg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random_state}"
        model_info = ModelInfo(
            model_id=model_id,
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            feature_names=feature_columns,
            target_name=target_column,
        )

        # Cache model
        self._model_cache[model_id] = {
            "sklearn_model": sklearn_model,
            "sm_model": sm_model,
            "info": model_info,
        }

        # Serialize model
        await self.save_model(model_id)

        logger.info(
            f"Model trained successfully: {model_id}, R²={r2:.4f}, "
            f"duration={metrics.training_duration_seconds:.2f}s"
        )

        return sklearn_model, metrics, model_info

    async def save_model(self, model_id: str) -> Path:
        """Serialize model to disk using joblib."""
        if model_id not in self._model_cache:
            raise ValueError(f"Model {model_id} not found in cache")

        model_data = self._model_cache[model_id]
        model_path = self.model_dir / f"{model_id}.joblib"

        # Save both models and metadata
        payload = {
            "sklearn_model": model_data["sklearn_model"],
            "sm_model": model_data["sm_model"],
            "info": model_data["info"].model_dump(),
        }

        # Run serialization in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None, joblib.dump, payload, model_path
        )

        logger.info(f"Model saved to {model_path}")
        return model_path

    async def load_model(self, model_id: str) -> Any:
        """Load a serialized model from disk."""
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Run deserialization in thread pool
        payload = await asyncio.get_event_loop().run_in_executor(
            None, joblib.load, model_path
        )

        self._model_cache[model_id] = payload
        logger.info(f"Model loaded from {model_path}")
        return payload["sklearn_model"]

    async def predict(self, model_id: str, features: Dict[str, float]) -> float:
        """Make prediction using a trained model."""
        if model_id not in self._model_cache:
            await self.load_model(model_id)

        model_data = self._model_cache[model_id]
        model = model_data["sklearn_model"]
        feature_names = model_data["info"]["feature_names"]

        # Validate features
        missing_features = set(feature_names) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Prepare feature vector
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)

        # Run prediction in thread pool
        prediction = await asyncio.get_event_loop().run_in_executor(
            None, model.predict, feature_vector
        )

        return float(prediction[0])


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0",
)

# Global model manager instance
model_manager = RegressionModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/train", response_model=TrainingMetrics)
async def train_endpoint(
    file: UploadFile = File(...),
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
    feature_columns: Optional[str] = None,
) -> TrainingMetrics:
    """
    Train a regression model from uploaded CSV data.

    Args:
        file: CSV file upload
        target_column: Name of target column
        test_size: Test split ratio (0.1-0.5)
        random_state: Random seed
        feature_columns: Comma-separated list of feature columns

    Returns:
        TrainingMetrics: Model training metrics
    """
    try:
        # Validate request parameters
        request = TrainingRequest(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            feature_columns=feature_columns.split(",") if feature_columns else None,
        )

        # Read and parse CSV
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        logger.info(
            f"Received training request: file={file.filename}, "
            f"rows={len(data)}, cols={len(data.columns)}"
        )

        # Train model
        try:
            _, metrics, _ = await model_manager.train_model(
                data=data,
                target_column=request.target_column,
                test_size=request.test_size,
                random_state=request.random_state,
                feature_columns=request.feature_columns,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return metrics

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during training")


@app.post("/predict/{model_id}")
async def predict_endpoint(
    model_id: str, features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Make predictions using a trained model.

    Args:
        model_id: ID of the trained model
        features: Dictionary of feature values

    Returns:
        Dict containing prediction result
    """
    try:
        prediction = await model_manager.predict(model_id, features)
        return {
            "model_id": model_id,
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    try:
        if model_id not in model_manager._model_cache:
            await model_manager.load_model(model_id)

        model_data = model_manager._model_cache[model_id]
        return model_data["info"]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all available models."""
    try:
        model_files = list(model_manager.model_dir.glob("*.joblib"))
        model_ids = [f.stem for f in model_files]
        return {"models": model_ids}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4
