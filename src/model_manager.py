import asyncio
import io
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger("regression_api")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_format)

# File handler with rotation
file_handler = RotatingFileHandler(
    log_dir / "regression_api.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context)s"
)
file_handler.setFormatter(file_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""

    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""

    target_column: str = Field(..., min_length=1, description="Name of target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test split ratio")
    random_state: int = Field(42, ge=0, description="Random seed")
    feature_columns: Optional[List[str]] = Field(
        None, description="List of feature columns (default: all except target)"
    )

    @validator("target_column")
    def validate_target_column(cls, v):
        if not v or not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()

    @validator("feature_columns")
    def validate_feature_columns(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("Feature columns list cannot be empty")
            # Remove duplicates while preserving order
            seen = set()
            unique_cols = []
            for col in v:
                if col not in seen:
                    seen.add(col)
                    unique_cols.append(col)
            return unique_cols
        return v


class TrainingResponse(BaseModel):
    """Response model for training results."""

    model_id: str
    training_timestamp: datetime
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    feature_importance: Dict[str, float]
    model_path: str
    data_shape: Dict[str, int]
    training_duration: float


class ModelManager:
    """Manages regression model training, evaluation, and serialization."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        self._model_cache: Dict[str, Any] = {}
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        request: TrainingRequest,
    ) -> TrainingResponse:
        """Train a regression model with comprehensive diagnostics."""
        start_time = time.time()
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

        try:
            # Validate data
            await self._validate_data(data, request)

            # Prepare features and target
            X, y, feature_names = await self._prepare_data(data, request)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=request.test_size,
                random_state=request.random_state,
            )

            # Train scikit-learn model
            sklearn_model = LinearRegression()
            sklearn_model.fit(X_train, y_train)

            # Train statsmodels for statistical inference
            X_train_sm = sm.add_constant(X_train)
            statsmodels_model = sm.OLS(y_train, X_train_sm).fit()

            # Make predictions
            y_pred_train = sklearn_model.predict(X_train)
            y_pred_test = sklearn_model.predict(X_test)

            # Calculate metrics
            metrics = await self._calculate_metrics(
                y_train, y_pred_train, y_test, y_pred_test
            )

            # Extract coefficients and p-values
            coefficients = dict(
                zip(feature_names, sklearn_model.coef_.tolist())
            )
            p_values = dict(
                zip(feature_names, statsmodels_model.pvalues[1:].tolist())
            )

            # Calculate feature importance (absolute coefficient values)
            feature_importance = {
                name: abs(coef) for name, coef in coefficients.items()
            }

            # Normalize feature importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {
                    name: importance / total_importance
                    for name, importance in feature_importance.items()
                }

            # Prepare model bundle
            model_bundle = {
                "sklearn_model": sklearn_model,
                "statsmodels_model": statsmodels_model,
                "feature_names": feature_names,
                "target_column": request.target_column,
                "model_id": model_id,
                "training_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "test_size": request.test_size,
                    "random_state": request.random_state,
                    "n_samples": len(data),
                    "n_features": len(feature_names),
                },
            }

            # Serialize model
            model_path = await self._save_model(model_bundle, model_id)

            # Cache model
            async with self._lock:
                self._model_cache[model_id] = model_bundle

            training_duration = time.time() - start_time

            logger.info(
                f"Model trained successfully",
                extra={
                    "context": {
                        "model_id": model_id,
                        "duration": training_duration,
                        "r_squared": metrics["r_squared_test"],
                        "n_samples": len(data),
                    }
                },
            )

            return TrainingResponse(
                model_id=model_id,
                training_timestamp=datetime.now(),
                metrics=metrics,
                coefficients=coefficients,
                p_values=p_values,
                r_squared=metrics["r_squared_test"],
                adjusted_r_squared=metrics["adjusted_r_squared_test"],
                feature_importance=feature_importance,
                model_path=str(model_path),
                data_shape={"rows": len(data), "columns": len(data.columns)},
                training_duration=training_duration,
            )

        except Exception as e:
            logger.error(
                f"Model training failed: {str(e)}",
                extra={"context": {"model_id": model_id}},
            )
            raise ModelTrainingError(f"Model training failed: {str(e)}")

    async def _validate_data(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> None:
        """Validate input data."""
        if data.empty:
            raise DataValidationError("Data is empty")

        if request.target_column not in data.columns:
            raise DataValidationError(
                f"Target column '{request.target_column}' not found in data"
            )

        # Check for non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise DataValidationError(
                f"Non-numeric columns found: {non_numeric}. Please encode categorical variables."
            )

        # Check for missing values
        if data.isnull().any().any():
            raise DataValidationError(
                "Data contains missing values. Please handle missing data before training."
            )

        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            raise DataValidationError(
                "Data contains infinite values. Please handle infinite values before training."
            )

        # Validate feature columns if provided
        if request.feature_columns:
            missing_cols = set(request.feature_columns) - set(data.columns)
            if missing_cols:
                raise DataValidationError(
                    f"Feature columns not found in data: {missing_cols}"
                )

    async def _prepare_data(
        self, data: pd.DataFrame, request: TrainingRequest
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training."""
        target = data[request.target_column]

        if request.feature_columns:
            feature_names = request.feature_columns
            features = data[feature_names]
        else:
            feature_names = [
                col for col in data.columns if col != request.target_column
            ]
            features = data[feature_names]

        # Ensure all features are numeric
        features = features.astype(float)
        target = target.astype(float)

        return (
            features.values,
            target.values,
            feature_names,
        )

    async def _calculate_metrics(
        self,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        # Training metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        # Test metrics
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        # Adjusted R² for test set
        n_test = len(y_test)
        p_test = 1  # number of predictors (simplified)
        adjusted_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (
            n_test - p_test - 1
        )

        return {
            "mse_train": float(mse_train),
            "rmse_train": float(rmse_train),
            "mae_train": float(mae_train),
            "r_squared_train": float(r2_train),
            "mse_test": float(mse_test),
            "rmse_test": float(rmse_test),
            "mae_test": float(mae_test),
            "r_squared_test": float(r2_test),
            "adjusted_r_squared_test": float(adjusted_r2_test),
        }

    async def _save_model(self, model_bundle: Dict[str, Any], model_id: str) -> Path:
        """Serialize model to disk."""
        model_path = self.model_dir / f"{model_id}.joblib"
        try:
            # Use thread pool for CPU-bound serialization
            await asyncio.to_thread(
                joblib.dump, model_bundle, model_path, compress=3
            )
            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ModelTrainingError(f"Failed to save model: {str(e)}")

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a serialized model."""
        # Check cache first
        async with self._lock:
            if model_id in self._model_cache:
                return self._model_cache[model_id]

        # Load from disk
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found")

        try:
            model_bundle = await asyncio.to_thread(joblib.load, model_path)
            async with self._lock:
                self._model_cache[model_id] = model_bundle
            return model_bundle
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelTrainingError(f"Failed to load model: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0",
)

# Initialize model manager
model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=TrainingResponse)
async def train_model(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of target column"),
    test_size: float = File(0.2, description="Test split ratio (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(
        None, description="Comma-separated list of feature columns"
    ),
) -> TrainingResponse:
    """Train a regression model from uploaded CSV data."""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    logger.info(
        f"Training request received",
        extra={"context": {"request_id": request_id, "filename": file.filename}},
    )

    try:
        # Validate file extension
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400, detail="Only CSV files are supported"
            )

        # Read and parse CSV data
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse CSV file: {str(e)}"
            )

        # Parse feature columns if provided
        feature_cols = None
        if feature_columns:
            feature_cols = [
                col.strip() for col in feature_columns.split(",") if col.strip()
            ]

        # Create training request
        try:
            training_request = TrainingRequest(
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                feature_columns=feature_cols,
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid training parameters: {str(e)}"
            )

        # Train model
        try:
            response = await model_manager.train_model(data, training_request)
            return response
        except DataValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except ModelTrainingError as e:
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during training",
            extra={"context": {"request_id": request_id, "error": str(e)}},
        )
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    try:
        model_bundle = await model_manager.load_model(model_id)
        return {
            "model_id": model_id,
            "feature_names": model_bundle["feature_names"],
            "target_column": model_bundle["target_column"],
            "training_metadata": model_bundle["training_metadata"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/predict/{model_id}")
async def predict(
    model_id: str,
    file: UploadFile = File(..., description="CSV file with features for prediction"),
) -> Dict[str, Any]:
    """Make predictions using a trained model."""
    try:
        # Load model
        model_bundle = await model_manager.load_model(model_id)

        # Read prediction data
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse CSV file: {str(e)}"
            )

        # Validate features
        feature_names = model_bundle["feature_names"]
        missing_cols = set(feature_names) - set(data.columns)
        if missing_cols:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required features: {missing_cols}",
            )

        # Prepare features
        X = data[feature_names].astype(float)

        # Make predictions
        sklearn_model = model_bundle["sklearn_model"]
        predictions = await asyncio.to_thread(sklearn_model.predict, X)

        return {
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "n_predictions": len(predictions),
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5
