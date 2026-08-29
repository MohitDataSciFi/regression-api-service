import asyncio
import io
import logging
import tempfile
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
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ==================== Schemas ====================
class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns. If None, all except target are used")

    @validator("target_column")
    def validate_target_column(cls, v):
        if not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()

    @validator("feature_columns")
    def validate_feature_columns(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("Feature columns list cannot be empty")
            if len(v) != len(set(v)):
                raise ValueError("Feature columns must be unique")
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
    mae: float
    rmse: float
    training_samples: int
    test_samples: int
    feature_columns: List[str]
    target_column: str
    created_at: datetime
    model_path: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: datetime


# ==================== Model Manager ====================
class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_state: int,
        feature_columns: Optional[List[str]] = None
    ) -> TrainingResponse:
        """Train a regression model with full diagnostics."""
        async with self._lock:
            try:
                logger.info(f"Starting model training with target: {target_column}")
                
                # Validate data
                if data.empty:
                    raise ValueError("DataFrame is empty")
                
                if target_column not in data.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                
                # Determine feature columns
                if feature_columns is None:
                    feature_columns = [col for col in data.columns if col != target_column]
                else:
                    # Validate feature columns exist
                    missing_cols = set(feature_columns) - set(data.columns)
                    if missing_cols:
                        raise ValueError(f"Missing feature columns: {missing_cols}")
                
                if not feature_columns:
                    raise ValueError("No feature columns available for training")
                
                # Prepare data
                X = data[feature_columns].copy()
                y = data[target_column].copy()
                
                # Check for non-numeric data
                if not np.issubdtype(X.dtypes.values[0], np.number):
                    raise ValueError("Feature columns must be numeric")
                if not np.issubdtype(y.dtype, np.number):
                    raise ValueError("Target column must be numeric")
                
                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    logger.warning("Missing values detected, dropping rows")
                    mask = X.notnull().all(axis=1) & y.notnull()
                    X = X[mask]
                    y = y[mask]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train sklearn model
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)
                
                # Train statsmodels for diagnostics
                X_train_const = add_constant(X_train)
                stats_model = OLS(y_train, X_train_const).fit()
                
                # Predictions
                y_pred_train = sklearn_model.predict(X_train)
                y_pred_test = sklearn_model.predict(X_test)
                
                # Compute metrics
                r_squared = r2_score(y_test, y_pred_test)
                mse = mean_squared_error(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                
                # Adjusted R-squared
                n = len(y_test)
                p = len(feature_columns)
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
                
                # Extract coefficients and p-values
                coefficients = dict(zip(feature_columns, sklearn_model.coef_))
                coefficients["intercept"] = float(sklearn_model.intercept_)
                
                p_values = {}
                for i, col in enumerate(feature_columns):
                    p_values[col] = float(stats_model.pvalues[i + 1])  # +1 for constant
                p_values["intercept"] = float(stats_model.pvalues[0])
                
                # Additional diagnostics
                residuals = y_test - y_pred_test
                dw_stat = durbin_watson(residuals)
                bp_test = het_breuschpagan(residuals, add_constant(X_test))
                
                # Create model ID
                model_id = f"reg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
                
                # Serialize model
                model_path = self.model_dir / f"{model_id}.joblib"
                model_data = {
                    "model": sklearn_model,
                    "stats_model": stats_model,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "metrics": {
                        "r_squared": r_squared,
                        "adjusted_r_squared": adjusted_r_squared,
                        "mse": mse,
                        "mae": mae,
                        "rmse": rmse,
                        "durbin_watson": float(dw_stat),
                        "breusch_pagan_lm": float(bp_test[0]),
                        "breusch_pagan_pvalue": float(bp_test[1])
                    },
                    "coefficients": coefficients,
                    "p_values": p_values,
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "created_at": datetime.now().isoformat()
                }
                
                joblib.dump(model_data, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Build response
                response = TrainingResponse(
                    model_id=model_id,
                    metrics={
                        "r_squared": r_squared,
                        "adjusted_r_squared": adjusted_r_squared,
                        "mse": mse,
                        "mae": mae,
                        "rmse": rmse,
                        "durbin_watson": float(dw_stat),
                        "breusch_pagan_lm": float(bp_test[0]),
                        "breusch_pagan_pvalue": float(bp_test[1])
                    },
                    coefficients=coefficients,
                    p_values=p_values,
                    r_squared=r_squared,
                    adjusted_r_squared=adjusted_r_squared,
                    mse=mse,
                    mae=mae,
                    rmse=rmse,
                    training_samples=len(X_train),
                    test_samples=len(X_test),
                    feature_columns=feature_columns,
                    target_column=target_column,
                    created_at=datetime.now(),
                    model_path=str(model_path)
                )
                
                logger.info(f"Training completed successfully. Model ID: {model_id}")
                return response
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a serialized model."""
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        try:
            model_data = joblib.load(model_path)
            logger.info(f"Model {model_id} loaded successfully")
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    async def predict(self, model_id: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        model_data = await self.load_model(model_id)
        
        try:
            # Validate features
            missing_features = set(model_data["feature_columns"]) - set(features.keys())
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Prepare feature vector
            feature_vector = np.array([features[col] for col in model_data["feature_columns"]]).reshape(1, -1)
            
            # Make prediction
            prediction = model_data["model"].predict(feature_vector)[0]
            
            return {
                "model_id": model_id,
                "prediction": float(prediction),
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# ==================== API ====================
app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0"
)

model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=TrainingResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
) -> TrainingResponse:
    """Train a regression model from uploaded CSV data."""
    try:
        # Validate request parameters
        request_data = {
            "target_column": target_column,
            "test_size": test_size,
            "random_state": random_state,
            "feature_columns": feature_columns.split(",") if feature_columns else None
        }
        
        try:
            validated_request = TrainingRequest(**request_data)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Read and parse CSV
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to parse CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        
        # Train model
        response = await model_manager.train_model(
            data=data,
            target_column=validated_request.target_column,
            test_size=validated_request.test_size,
            random_state=validated_request.random_state,
            feature_columns=validated_request.feature_columns
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/{model_id}")
async def predict_endpoint(model_id: str, features: Dict[str, float]) -> Dict[str, Any]:
    """Make predictions using a trained model."""
    return await model_manager.predict(model_id, features)


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    model_data = await model_manager.load_model(model_id)
    return {
        "model_id": model_id,
        "feature_columns": model_data["feature_columns"],
        "target_column": model_data["target_column"],
        "metrics": model_data["metrics"],
        "coefficients": model_data["coefficients"],
        "p_values": model_data["p_values"],
        "training_samples": model_data["training_samples"],
        "test_samples": model_data["test_samples"],
        "created_at": model_data["created_at"]
    }


@app.get("/models")
async def list_models() -> List[Dict[str, Any]]:
    """List all trained models."""
    models = []
    for model_file in model_manager.model_dir.glob("*.joblib"):
        try:
            model_data = joblib.load(model_file)
            models.append({
                "model_id": model_file.stem,
                "target_column": model_data["target_column"],
                "feature_columns": model_data["feature_columns"],
                "r_squared": model_data["metrics"]["r_squared"],
                "created_at": model_data["created_at"]
            })
        except Exception as e:
            logger.error(f"Failed to load model info for {model_file}: {str(e)}")
    
    return models


# ==================== Error Handlers ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return {
        "error": "HTTPException",
        "detail": exc.detail,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "InternalServerError",
        "detail": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5

# Phase 1: Core Model Training and Serialization - iteration 6

# Phase 1: Core Model Training and Serialization - iteration 7
