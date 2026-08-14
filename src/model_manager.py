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

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""

    target_column: str = Field(..., min_length=1, description="Target column name")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed")
    feature_columns: Optional[List[str]] = Field(
        None, description="Feature columns to use (default: all except target)"
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


class TrainingResponse(BaseModel):
    """Response model for training results."""

    model_id: str
    training_timestamp: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    feature_importance: Dict[str, float]
    model_path: str
    data_shape: Dict[str, int]


class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self._model_cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Train a regression model with diagnostics and serialization."""
        start_time = time.time()
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"

        try:
            # Validate data
            if data.empty:
                raise DataValidationError("Data cannot be empty")

            if target_column not in data.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in data",
                    {"available_columns": list(data.columns)},
                )

            # Prepare features
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            else:
                missing_cols = set(feature_columns) - set(data.columns)
                if missing_cols:
                    raise DataValidationError(
                        f"Missing feature columns: {missing_cols}",
                        {"missing_columns": list(missing_cols)},
                    )

            if len(feature_columns) == 0:
                raise DataValidationError("No feature columns available for training")

            # Extract data
            X = data[feature_columns].copy()
            y = data[target_column].copy()

            # Check for non-numeric data
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                raise DataValidationError(
                    f"Non-numeric columns found: {non_numeric_cols}",
                    {"non_numeric_columns": non_numeric_cols},
                )

            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning(f"Missing values detected in data for model {model_id}")
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Train sklearn model
            sklearn_model = LinearRegression()
            sklearn_model.fit(X_train, y_train)

            # Train statsmodels for diagnostics
            X_train_sm = sm.add_constant(X_train)
            statsmodels_model = sm.OLS(y_train, X_train_sm).fit()

            # Compute metrics
            y_pred = sklearn_model.predict(X_test)
            metrics = {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mean_absolute_error": float(mean_absolute_error(y_test, y_pred)),
                "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
                "root_mean_squared_error": float(
                    np.sqrt(mean_squared_error(y_test, y_pred))
                ),
                "training_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "training_time_seconds": float(time.time() - start_time),
            }

            # Extract coefficients and p-values
            coefficients = {}
            p_values = {}
            feature_importance = {}

            for i, col in enumerate(feature_columns):
                coefficients[col] = float(sklearn_model.coef_[i])
                p_values[col] = float(statsmodels_model.pvalues[i + 1])  # +1 for constant
                # Simple feature importance based on absolute coefficient * std
                feature_importance[col] = float(
                    abs(sklearn_model.coef_[i]) * X[col].std()
                )

            # Normalize feature importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {
                    k: v / total_importance for k, v in feature_importance.items()
                }

            # Prepare model package
            model_package = {
                "sklearn_model": sklearn_model,
                "statsmodels_model": statsmodels_model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "model_id": model_id,
                "training_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "test_size": test_size,
                    "random_state": random_state,
                    "data_shape": {"rows": len(data), "columns": len(data.columns)},
                },
            }

            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            await asyncio.to_thread(joblib.dump, model_package, model_path)

            # Cache model
            async with self._lock:
                self._model_cache[model_id] = model_package

            # Prepare response
            response = {
                "model_id": model_id,
                "training_timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "coefficients": coefficients,
                "p_values": p_values,
                "r_squared": float(statsmodels_model.rsquared),
                "adjusted_r_squared": float(statsmodels_model.rsquared_adj),
                "feature_importance": feature_importance,
                "model_path": str(model_path),
                "data_shape": {"rows": len(data), "columns": len(data.columns)},
            }

            logger.info(
                f"Model trained successfully: {model_id}",
                extra={
                    "context": {
                        "model_id": model_id,
                        "r2_score": metrics["r2_score"],
                        "training_time": metrics["training_time_seconds"],
                    }
                },
            )

            return response, model_id

        except DataValidationError as e:
            logger.error(f"Data validation error: {e.message}", extra={"context": e.details})
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            logger.error(
                f"Model training failed: {str(e)}",
                extra={"context": {"model_id": model_id}},
            )
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a serialized model from cache or disk."""
        async with self._lock:
            if model_id in self._model_cache:
                return self._model_cache[model_id]

        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        try:
            model_package = await asyncio.to_thread(joblib.load, model_path)
            async with self._lock:
                self._model_cache[model_id] = model_package
            logger.info(f"Model loaded from disk: {model_id}")
            return model_package
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    async def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        model_package = await self.load_model(model_id)
        sklearn_model = model_package["sklearn_model"]
        feature_columns = model_package["feature_columns"]

        # Validate features
        missing_cols = set(feature_columns) - set(data.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature columns: {missing_cols}",
            )

        X = data[feature_columns]
        predictions = await asyncio.to_thread(sklearn_model.predict, X)
        return predictions


# Initialize FastAPI app
app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0",
)

# Initialize model manager
model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=TrainingResponse)
async def train_model_endpoint(
    file: UploadFile = File(..., description="CSV file with training data"),
    target_column: str = File(..., description="Target column name"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(
        None, description="Comma-separated feature columns"
    ),
) -> TrainingResponse:
    """Train a regression model from uploaded CSV data."""
    request_start = time.time()
    logger.info(
        f"Training request received",
        extra={
            "context": {
                "filename": file.filename,
                "target_column": target_column,
                "test_size": test_size,
                "random_state": random_state,
            }
        },
    )

    try:
        # Validate request parameters
        try:
            training_request = TrainingRequest(
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                feature_columns=(
                    feature_columns.split(",") if feature_columns else None
                ),
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

        # Read and parse CSV data
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid CSV format: {str(e)}"
            )

        # Train model
        response, model_id = await model_manager.train_model(
            data=data,
            target_column=training_request.target_column,
            test_size=training_request.test_size,
            random_state=training_request.random_state,
            feature_columns=training_request.feature_columns,
        )

        # Add request processing time
        response["metrics"]["total_request_time_seconds"] = time.time() - request_start

        logger.info(
            f"Training completed successfully",
            extra={
                "context": {
                    "model_id": model_id,
                    "total_time": response["metrics"]["total_request_time_seconds"],
                }
            },
        )

        return TrainingResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/{model_id}")
async def predict_endpoint(
    model_id: str,
    file: UploadFile = File(..., description="CSV file with prediction data"),
) -> Dict[str, Any]:
    """Make predictions using a trained model."""
    logger.info(f"Prediction request received for model: {model_id}")

    try:
        # Read and parse CSV data
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid CSV format: {str(e)}"
            )

        # Make predictions
        predictions = await model_manager.predict(model_id, data)

        # Prepare response
        response = {
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat(),
            "num_predictions": len(predictions),
        }

        logger.info(
            f"Predictions generated successfully",
            extra={"context": {"model_id": model_id, "num_predictions": len(predictions)}},
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    try:
        model_package = await model_manager.load_model(model_id)
        return {
            "model_id": model_id,
            "feature_columns": model_package["feature_columns"],
            "target_column": model_package["target_column"],
            "training_metadata": model_package["training_metadata"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all available models."""
    try:
        model_files = list(model_manager.model_dir.glob("*.joblib"))
        model_ids = [f.stem for f in model_files]
        return {"models": model_ids, "count": len(model_ids)}
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4
