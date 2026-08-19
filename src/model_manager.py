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
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError, validator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("regression_api.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_EXTENSIONS = {".csv", ".txt"}


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns to use")

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
            if len(v) != len(set(v)):
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
    f_statistic: float
    f_p_value: float
    mse: float
    rmse: float
    mae: float
    n_samples: int
    n_features: int
    feature_columns: List[str]
    target_column: str
    model_path: str


class ModelManager:
    """Manages regression model training, serialization, and diagnostics."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        self._models: Dict[str, Dict[str, Any]] = {}

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train a regression model with full diagnostics."""
        start_time = time.time()
        logger.info(f"Starting model training with target: {target_column}")

        try:
            # Validate data
            if data.empty:
                raise ValueError("Data cannot be empty")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            # Prepare features
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            else:
                missing_cols = set(feature_columns) - set(data.columns)
                if missing_cols:
                    raise ValueError(f"Missing feature columns: {missing_cols}")

            # Validate feature columns
            for col in feature_columns:
                if not np.issubdtype(data[col].dtype, np.number):
                    raise ValueError(f"Feature column '{col}' must be numeric")

            if not np.issubdtype(data[target_column].dtype, np.number):
                raise ValueError(f"Target column '{target_column}' must be numeric")

            # Remove rows with missing values
            data_clean = data[feature_columns + [target_column]].dropna()
            if len(data_clean) < 10:
                raise ValueError("Insufficient data after removing missing values")

            X = data_clean[feature_columns].values
            y = data_clean[target_column].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Train scikit-learn model
            sklearn_model = LinearRegression()
            sklearn_model.fit(X_train, y_train)

            # Predictions
            y_pred_train = sklearn_model.predict(X_train)
            y_pred_test = sklearn_model.predict(X_test)

            # Calculate metrics
            metrics = {
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_mse": mean_squared_error(y_train, y_pred_train),
                "test_mse": mean_squared_error(y_test, y_pred_test),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test)
            }

            # Statsmodels for diagnostics
            X_with_const = add_constant(X)
            stats_model = OLS(y, X_with_const).fit()

            # Extract coefficients and p-values
            coefficients = dict(zip(feature_columns, sklearn_model.coef_))
            coefficients["intercept"] = float(sklearn_model.intercept_)

            p_values = {}
            for i, col in enumerate(feature_columns):
                p_values[col] = float(stats_model.pvalues[i + 1])  # +1 for constant
            p_values["intercept"] = float(stats_model.pvalues[0])

            # Additional diagnostics
            residuals = stats_model.resid
            dw_statistic = durbin_watson(residuals)
            try:
                bp_test = het_breuschpagan(residuals, X_with_const)
                bp_statistic = float(bp_test[0])
                bp_p_value = float(bp_test[1])
            except Exception as e:
                logger.warning(f"Breusch-Pagan test failed: {e}")
                bp_statistic = float("nan")
                bp_p_value = float("nan")

            # Generate model ID
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"

            # Prepare model artifact
            model_artifact = {
                "model_id": model_id,
                "sklearn_model": sklearn_model,
                "statsmodels_model": stats_model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "coefficients": coefficients,
                "p_values": p_values,
                "metrics": metrics,
                "training_params": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "n_samples": len(data_clean),
                    "n_features": len(feature_columns)
                },
                "diagnostics": {
                    "dw_statistic": float(dw_statistic),
                    "bp_statistic": bp_statistic,
                    "bp_p_value": bp_p_value
                },
                "training_timestamp": datetime.now().isoformat(),
                "training_duration": time.time() - start_time
            }

            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            await self._save_model(model_artifact, model_path)

            # Store in memory
            async with self._lock:
                self._models[model_id] = model_artifact

            logger.info(f"Model trained successfully: {model_id} in {time.time() - start_time:.2f}s")

            # Build response
            response = TrainingResponse(
                model_id=model_id,
                training_timestamp=model_artifact["training_timestamp"],
                metrics=metrics,
                coefficients=coefficients,
                p_values=p_values,
                r_squared=float(stats_model.rsquared),
                adjusted_r_squared=float(stats_model.rsquared_adj),
                f_statistic=float(stats_model.fvalue),
                f_p_value=float(stats_model.f_pvalue),
                mse=metrics["test_mse"],
                rmse=metrics["test_rmse"],
                mae=metrics["test_mae"],
                n_samples=len(data_clean),
                n_features=len(feature_columns),
                feature_columns=feature_columns,
                target_column=target_column,
                model_path=str(model_path)
            )

            return response.dict()

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    async def _save_model(self, model_artifact: Dict[str, Any], path: Path) -> None:
        """Save model artifact to disk asynchronously."""
        try:
            # Use asyncio.to_thread for CPU-bound serialization
            await asyncio.to_thread(joblib.dump, model_artifact, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model from disk or memory."""
        # Check memory first
        async with self._lock:
            if model_id in self._models:
                return self._models[model_id]

        # Try loading from disk
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        try:
            model_artifact = await asyncio.to_thread(joblib.load, model_path)
            async with self._lock:
                self._models[model_id] = model_artifact
            return model_artifact
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    async def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        model_artifact = await self.load_model(model_id)
        sklearn_model = model_artifact["sklearn_model"]
        feature_columns = model_artifact["feature_columns"]

        # Validate features
        missing_cols = set(feature_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        X = data[feature_columns].values
        predictions = await asyncio.to_thread(sklearn_model.predict, X)
        return predictions


# Initialize FastAPI app and model manager
app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0"
)
model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=TrainingResponse)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file with training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
) -> Dict[str, Any]:
    """Train a regression model from uploaded CSV data."""
    start_time = time.time()
    logger.info(f"Received training request for target: {target_column}")

    # Validate request parameters
    try:
        request = TrainingRequest(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            feature_columns=feature_columns.split(",") if feature_columns else None
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=e.errors())

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )

    # Parse CSV data
    try:
        data = pd.read_csv(io.BytesIO(contents))
        logger.info(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    # Train model
    try:
        result = await model_manager.train_model(
            data=data,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            feature_columns=request.feature_columns
        )
        logger.info(f"Training completed in {time.time() - start_time:.2f}s")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict/{model_id}")
async def predict_endpoint(
    model_id: str,
    file: UploadFile = File(..., description="CSV file with features for prediction")
) -> Dict[str, Any]:
    """Make predictions using a trained model."""
    logger.info(f"Prediction request for model: {model_id}")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )

    # Parse CSV data
    try:
        data = pd.read_csv(io.BytesIO(contents))
        logger.info(f"Loaded prediction data with {len(data)} rows")
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    # Make predictions
    try:
        predictions = await model_manager.predict(model_id, data)
        return {
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "n_predictions": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    try:
        model_artifact = await model_manager.load_model(model_id)
        return {
            "model_id": model_artifact["model_id"],
            "feature_columns": model_artifact["feature_columns"],
            "target_column": model_artifact["target_column"],
            "metrics": model_artifact["metrics"],
            "coefficients": model_artifact["coefficients"],
            "p_values": model_artifact["p_values"],
            "training_timestamp": model_artifact["training_timestamp"],
            "training_duration": model_artifact["training_duration"],
            "diagnostics": model_artifact["diagnostics"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all available models."""
    model_files = list(self.model_dir.glob("*.joblib")) if hasattr(self, 'model_dir') else []
    model_ids = [f.stem for f in model_files]
    return {"models": model_ids}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3
