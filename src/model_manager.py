import asyncio
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        logging.FileHandler("regression_service.log"),
        logging.StreamHandler()
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
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns (default: all except target)")

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
            for col in v:
                if not col.strip():
                    raise ValueError("Feature column names cannot be empty")
        return v


class TrainingResponse(BaseModel):
    """Response model for training results."""
    model_id: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    mse: float
    rmse: float
    mae: float
    training_samples: int
    test_samples: int
    feature_count: int
    timestamp: str
    model_path: str


class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""

    def __init__(self) -> None:
        self.model_dir = MODEL_DIR
        self._lock = asyncio.Lock()
        logger.info("ModelManager initialized with model directory: %s", self.model_dir)

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_state: int,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """Train an OLS regression model with diagnostics."""
        async with self._lock:
            try:
                logger.info("Starting model training with target: %s, test_size: %.2f", target_column, test_size)

                # Validate data
                if data.empty:
                    raise ValueError("DataFrame is empty")
                if target_column not in data.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                if not pd.api.types.is_numeric_dtype(data[target_column]):
                    raise ValueError(f"Target column '{target_column}' must be numeric")

                # Determine feature columns
                if feature_columns is None:
                    feature_columns = [col for col in data.columns if col != target_column]
                else:
                    missing_cols = set(feature_columns) - set(data.columns)
                    if missing_cols:
                        raise ValueError(f"Missing feature columns: {missing_cols}")

                # Validate feature columns are numeric
                for col in feature_columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise ValueError(f"Feature column '{col}' must be numeric")

                # Remove rows with missing values
                data_clean = data[[target_column] + feature_columns].dropna()
                if len(data_clean) < 10:
                    raise ValueError("Not enough data after removing missing values (minimum 10 rows)")

                # Split data
                X = data_clean[feature_columns].values
                y = data_clean[target_column].values

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                # Train scikit-learn model
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)

                # Train statsmodels for diagnostics
                X_train_const = add_constant(X_train)
                stats_model = OLS(y_train, X_train_const).fit()

                # Compute predictions and metrics
                y_pred_train = sklearn_model.predict(X_train)
                y_pred_test = sklearn_model.predict(X_test)

                # Calculate metrics
                r_squared = r2_score(y_test, y_pred_test)
                mse = mean_squared_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred_test)

                # Extract coefficients and p-values
                coefficients = dict(zip(feature_columns, sklearn_model.coef_))
                coefficients["intercept"] = float(sklearn_model.intercept_)

                p_values = {}
                for i, col in enumerate(feature_columns):
                    p_values[col] = float(stats_model.pvalues[i + 1])  # +1 for constant
                p_values["intercept"] = float(stats_model.pvalues[0])

                # Additional diagnostics
                adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - len(feature_columns) - 1)

                # Generate model ID and save
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
                model_path = self.model_dir / f"{model_id}.joblib"

                # Package model artifacts
                model_artifacts = {
                    "sklearn_model": sklearn_model,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "model_id": model_id,
                    "training_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "test_size": test_size,
                        "random_state": random_state,
                        "n_samples": len(data_clean),
                        "n_features": len(feature_columns)
                    }
                }

                # Serialize model
                joblib.dump(model_artifacts, model_path)
                logger.info("Model saved to %s", model_path)

                # Prepare response
                metrics = {
                    "r_squared": float(r_squared),
                    "adjusted_r_squared": float(adjusted_r_squared),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "train_r_squared": float(r2_score(y_train, y_pred_train)),
                    "train_mse": float(mean_squared_error(y_train, y_pred_train)),
                    "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                    "train_mae": float(mean_absolute_error(y_train, y_pred_train))
                }

                response_data = {
                    "model_id": model_id,
                    "metrics": metrics,
                    "coefficients": coefficients,
                    "p_values": p_values,
                    "r_squared": float(r_squared),
                    "adjusted_r_squared": float(adjusted_r_squared),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "feature_count": len(feature_columns),
                    "timestamp": datetime.now().isoformat(),
                    "model_path": str(model_path)
                }

                logger.info("Training completed successfully. Model ID: %s", model_id)
                return response_data, model_id

            except Exception as e:
                logger.error("Training failed: %s", str(e), exc_info=True)
                raise


class APIService:
    """FastAPI application for regression model training."""

    def __init__(self) -> None:
        self.app = FastAPI(
            title="Regression Model Training API",
            description="Production-grade API for training and serving regression models",
            version="1.0.0"
        )
        self.model_manager = ModelManager()
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health_check() -> Dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.post("/train", response_model=TrainingResponse)
        async def train_model(
            file: UploadFile = File(..., description="CSV file containing training data"),
            target_column: str = File(..., description="Name of the target column"),
            test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
            random_state: int = File(42, description="Random seed"),
            feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
        ) -> TrainingResponse:
            """Train a regression model from uploaded CSV data."""
            try:
                # Validate file
                if not file.filename:
                    raise HTTPException(status_code=400, detail="No file provided")

                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
                    )

                # Read and validate file size
                content = await file.read()
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024)}MB"
                    )

                # Parse CSV
                try:
                    data = pd.read_csv(io.BytesIO(content))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

                # Parse feature columns if provided
                feature_cols = None
                if feature_columns:
                    feature_cols = [col.strip() for col in feature_columns.split(",") if col.strip()]

                # Validate request parameters
                try:
                    request = TrainingRequest(
                        target_column=target_column,
                        test_size=test_size,
                        random_state=random_state,
                        feature_columns=feature_cols
                    )
                except ValidationError as e:
                    raise HTTPException(status_code=422, detail=str(e))

                # Train model
                try:
                    response_data, _ = await self.model_manager.train_model(
                        data=data,
                        target_column=request.target_column,
                        test_size=request.test_size,
                        random_state=request.random_state,
                        feature_columns=request.feature_columns
                    )
                    return TrainingResponse(**response_data)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception as e:
                    logger.error("Unexpected error during training: %s", str(e), exc_info=True)
                    raise HTTPException(status_code=500, detail="Internal server error during training")

        @self.app.get("/models/{model_id}")
        async def get_model_info(model_id: str) -> Dict[str, Any]:
            """Get information about a trained model."""
            model_path = self.model_manager.model_dir / f"{model_id}.joblib"
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

            try:
                model_artifacts = joblib.load(model_path)
                return {
                    "model_id": model_id,
                    "feature_columns": model_artifacts["feature_columns"],
                    "target_column": model_artifacts["target_column"],
                    "training_metadata": model_artifacts["training_metadata"],
                    "model_path": str(model_path)
                }
            except Exception as e:
                logger.error("Error loading model info: %s", str(e), exc_info=True)
                raise HTTPException(status_code=500, detail="Error loading model information")

        @self.app.get("/models")
        async def list_models() -> List[Dict[str, Any]]:
            """List all trained models."""
            models = []
            for model_file in self.model_manager.model_dir.glob("*.joblib"):
                try:
                    model_artifacts = joblib.load(model_file)
                    models.append({
                        "model_id": model_artifacts["model_id"],
                        "feature_columns": model_artifacts["feature_columns"],
                        "target_column": model_artifacts["target_column"],
                        "training_metadata": model_artifacts["training_metadata"]
                    })
                except Exception as e:
                    logger.warning("Failed to load model %s: %s", model_file.name, str(e))
            return models


# Create FastAPI app instance
api_service = APIService()
app = api_service.app


# Main entry point for running the service
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting regression API service...")
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4
