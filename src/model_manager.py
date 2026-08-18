import asyncio
import io
import logging
import time
from dataclasses import dataclass
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
        logging.FileHandler("regression_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Schemas (Pydantic models)
# ============================================================================

class TrainingRequest(BaseModel):
    """Validation schema for training request metadata."""
    target_column: str = Field(..., min_length=1, description="Name of the target variable")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test split ratio")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns. If None, all except target are used")

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
    """Response schema for training endpoint."""
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
    training_duration_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    detail: str
    timestamp: str


# ============================================================================
# Model Manager
# ============================================================================

@dataclass
class ModelArtifacts:
    """Container for model artifacts."""
    model_id: str
    sklearn_model: LinearRegression
    statsmodels_model: sm.OLS
    feature_names: List[str]
    target_name: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    feature_importance: Dict[str, float]
    model_path: str
    training_timestamp: str
    data_shape: Dict[str, int]
    training_duration: float


class ModelManager:
    """Manages regression model training, evaluation, and serialization."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelArtifacts] = {}
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None
    ) -> ModelArtifacts:
        """Train a regression model with comprehensive diagnostics."""
        start_time = time.time()
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        logger.info(f"Starting training for model {model_id}")

        try:
            # Validate data
            if data.empty:
                raise ValueError("Training data is empty")
            
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
            
            if len(feature_columns) == 0:
                raise ValueError("No feature columns available for training")
            
            # Extract data
            X = data[feature_columns].copy()
            y = data[target_column].copy()
            
            # Check for non-numeric data
            if not np.issubdtype(X.dtypes.iloc[0], np.number):
                raise ValueError("Feature columns must be numeric")
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("Target column must be numeric")
            
            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning("Missing values detected, dropping rows")
                mask = X.notnull().all(axis=1) & y.notnull()
                X = X[mask]
                y = y[mask]
            
            if len(X) < 10:
                raise ValueError("Insufficient data after cleaning (minimum 10 samples required)")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train sklearn model
            sklearn_model = LinearRegression()
            sklearn_model.fit(X_train, y_train)
            
            # Train statsmodels for p-values
            X_sm = sm.add_constant(X_train)
            statsmodels_model = sm.OLS(y_train, X_sm).fit()
            
            # Predictions
            y_pred_train = sklearn_model.predict(X_train)
            y_pred_test = sklearn_model.predict(X_test)
            
            # Compute metrics
            metrics = {
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_mse": mean_squared_error(y_train, y_pred_train),
                "test_mse": mean_squared_error(y_test, y_pred_test),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "n_samples": len(X),
                "n_features": len(feature_columns),
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test)
            }
            
            # Extract coefficients and p-values
            coefficients = dict(zip(feature_columns, sklearn_model.coef_))
            coefficients["intercept"] = sklearn_model.intercept_
            
            # Statsmodels p-values (excluding constant)
            p_values = {}
            for i, col in enumerate(feature_columns):
                p_values[col] = float(statsmodels_model.pvalues[i + 1])  # +1 for constant
            
            # R-squared and adjusted R-squared
            r_squared = statsmodels_model.rsquared
            adjusted_r_squared = statsmodels_model.rsquared_adj
            
            # Feature importance (absolute coefficient values normalized)
            abs_coefs = np.abs(sklearn_model.coef_)
            feature_importance = dict(zip(
                feature_columns,
                abs_coefs / abs_coefs.sum() if abs_coefs.sum() > 0 else abs_coefs
            ))
            
            # Create model artifacts
            training_duration = time.time() - start_time
            training_timestamp = datetime.now().isoformat()
            
            artifacts = ModelArtifacts(
                model_id=model_id,
                sklearn_model=sklearn_model,
                statsmodels_model=statsmodels_model,
                feature_names=feature_columns,
                target_name=target_column,
                metrics=metrics,
                coefficients=coefficients,
                p_values=p_values,
                r_squared=float(r_squared),
                adjusted_r_squared=float(adjusted_r_squared),
                feature_importance=feature_importance,
                model_path=str(self.model_dir / f"{model_id}.joblib"),
                training_timestamp=training_timestamp,
                data_shape={"rows": len(X), "columns": len(feature_columns) + 1},
                training_duration=training_duration
            )
            
            # Serialize model
            await self._save_model(artifacts)
            
            # Store in memory
            async with self._lock:
                self._models[model_id] = artifacts
            
            logger.info(f"Model {model_id} trained successfully in {training_duration:.2f}s")
            return artifacts
            
        except Exception as e:
            logger.error(f"Training failed for model {model_id}: {str(e)}")
            raise

    async def _save_model(self, artifacts: ModelArtifacts) -> None:
        """Serialize model artifacts to disk."""
        try:
            # Save in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                joblib.dump,
                {
                    "sklearn_model": artifacts.sklearn_model,
                    "statsmodels_model": artifacts.statsmodels_model,
                    "feature_names": artifacts.feature_names,
                    "target_name": artifacts.target_name,
                    "metrics": artifacts.metrics,
                    "coefficients": artifacts.coefficients,
                    "p_values": artifacts.p_values,
                    "r_squared": artifacts.r_squared,
                    "adjusted_r_squared": artifacts.adjusted_r_squared,
                    "feature_importance": artifacts.feature_importance,
                    "training_timestamp": artifacts.training_timestamp,
                    "data_shape": artifacts.data_shape,
                    "training_duration": artifacts.training_duration
                },
                artifacts.model_path
            )
            logger.info(f"Model saved to {artifacts.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    async def load_model(self, model_id: str) -> Optional[ModelArtifacts]:
        """Load a model from disk or memory."""
        # Check memory first
        async with self._lock:
            if model_id in self._models:
                return self._models[model_id]
        
        # Try loading from disk
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, joblib.load, str(model_path))
            
            artifacts = ModelArtifacts(
                model_id=model_id,
                sklearn_model=data["sklearn_model"],
                statsmodels_model=data["statsmodels_model"],
                feature_names=data["feature_names"],
                target_name=data["target_name"],
                metrics=data["metrics"],
                coefficients=data["coefficients"],
                p_values=data["p_values"],
                r_squared=data["r_squared"],
                adjusted_r_squared=data["adjusted_r_squared"],
                feature_importance=data["feature_importance"],
                model_path=str(model_path),
                training_timestamp=data["training_timestamp"],
                data_shape=data["data_shape"],
                training_duration=data["training_duration"]
            )
            
            async with self._lock:
                self._models[model_id] = artifacts
            
            logger.info(f"Model {model_id} loaded from disk")
            return artifacts
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None

    async def get_all_models(self) -> List[Dict[str, Any]]:
        """Get summary of all trained models."""
        async with self._lock:
            return [
                {
                    "model_id": m.model_id,
                    "training_timestamp": m.training_timestamp,
                    "r_squared": m.r_squared,
                    "adjusted_r_squared": m.adjusted_r_squared,
                    "n_features": len(m.feature_names),
                    "target": m.target_name
                }
                for m in self._models.values()
            ]


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0"
)

model_manager = ModelManager()


@app.post("/train", response_model=TrainingResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def train_model_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test split ratio (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
):
    """Train a regression model from uploaded CSV data."""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
    logger.info(f"[{request_id}] Training request received: target={target_column}, test_size={test_size}")
    
    try:
        # Validate request parameters
        try:
            request = TrainingRequest(
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                feature_columns=feature_columns.split(",") if feature_columns else None
            )
        except ValidationError as e:
            logger.warning(f"[{request_id}] Validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Read and parse CSV
        try:
            content = await file.read()
            data = pd.read_csv(io.BytesIO(content))
            logger.info(f"[{request_id}] Loaded data with shape: {data.shape}")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to parse CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Train model
        try:
            artifacts = await model_manager.train_model(
                data=data,
                target_column=request.target_column,
                test_size=request.test_size,
                random_state=request.random_state,
                feature_columns=request.feature_columns
            )
        except ValueError as e:
            logger.error(f"[{request_id}] Training error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected training error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
        
        # Build response
        response = TrainingResponse(
            model_id=artifacts.model_id,
            training_timestamp=artifacts.training_timestamp,
            metrics=artifacts.metrics,
            coefficients=artifacts.coefficients,
            p_values=artifacts.p_values,
            r_squared=artifacts.r_squared,
            adjusted_r_squared=artifacts.adjusted_r_squared,
            feature_importance=artifacts.feature_importance,
            model_path=artifacts.model_path,
            data_shape=artifacts.data_shape,
            training_duration_seconds=artifacts.training_duration
        )
        
        logger.info(f"[{request_id}] Training completed successfully: {artifacts.model_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models", response_model=List[Dict[str, Any]])
async def list_models():
    """List all trained models."""
    try:
        models = await model_manager.get_all_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{model_id}", response_model=TrainingResponse)
async def get_model(model_id: str):
    """Get details of a specific model."""
    try:
        artifacts = await model_manager.load_model(model_id)
        if artifacts is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return TrainingResponse(
            model_id=artifacts.model_id,
            training_timestamp=artifacts.training_timestamp,
            metrics=artifacts.metrics,
            coefficients=artifacts.coefficients,
            p_values=artifacts.p_values,
            r_squared=artifacts.r_squared,
            adjusted_r_squared=artifacts.adjusted_r_squared,
            feature_importance=artifacts.feature_importance,
            model_path=artifacts.model_path,
            data_shape=artifacts.data_shape,
            training_duration_seconds=artifacts.training_duration
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(await model_manager.get_all_models())
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Regression API Service")
    # Load any existing models from disk
    model_files = list(model_manager.model_dir.glob("*.joblib"))
    for model_file in model_files:
        model_id = model_file.stem
        try:
            await model_manager.load_model(model_id)
        except Exception as e:
            logger.error(f"Failed to load model {model_id} on startup: {str(e)}")
    logger.info(f"Loaded {len(model_files)} models from disk")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Regression API Service")
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5

# Phase 1: Core Model Training and Serialization - iteration 6

# Phase 1: Core Model Training and Serialization - iteration 7

# Phase 1: Core Model Training and Serialization - iteration 8

# Phase 1: Core Model Training and Serialization - iteration 9
