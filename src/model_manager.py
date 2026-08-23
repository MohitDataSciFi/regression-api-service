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
        logging.FileHandler("regression_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Schemas (Pydantic Models)
# =============================================================================

class TrainingRequest(BaseModel):
    """Validated training request parameters."""
    target_column: str = Field(..., min_length=1, description="Name of the target variable")
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
        return v


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""
    model_id: str
    training_timestamp: datetime
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    mse: float
    rmse: float
    mae: float
    durbin_watson: float
    breusch_pagan_pvalue: float
    feature_importance: Dict[str, float]
    model_path: str
    status: str = "success"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: datetime


# =============================================================================
# Model Manager
# =============================================================================

class RegressionModelManager:
    """Manages training, serialization, and diagnostics of regression models."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"Model manager initialized with directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Train an OLS regression model with comprehensive diagnostics.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            test_size: Proportion of test set
            random_state: Random seed
            feature_columns: Optional list of features to use
            
        Returns:
            Tuple of (metrics_dict, model_path)
        """
        start_time = time.time()
        logger.info(f"Starting model training with target={target_column}, test_size={test_size}")

        # Validate data
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        else:
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

        # Prepare data
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Handle missing values
        if X.isnull().any().any() or y.isnull().any():
            logger.warning("Missing values detected, dropping rows with NaN")
            mask = X.notnull().all(axis=1) & y.notnull()
            X = X[mask]
            y = y[mask]

        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

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
        X_train_const = add_constant(X_train)
        statsmodels_model = OLS(y_train, X_train_const).fit()

        # Predictions
        y_pred_train = sklearn_model.predict(X_train)
        y_pred_test = sklearn_model.predict(X_test)

        # Compute metrics
        metrics = {
            "r2_train": r2_score(y_train, y_pred_train),
            "r2_test": r2_score(y_test, y_pred_test),
            "mse_train": mean_squared_error(y_train, y_pred_train),
            "mse_test": mean_squared_error(y_test, y_pred_test),
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae_train": mean_absolute_error(y_train, y_pred_train),
            "mae_test": mean_absolute_error(y_test, y_pred_test),
            "training_time": time.time() - start_time,
            "n_samples": len(X),
            "n_features": len(feature_columns)
        }

        # Extract coefficients and p-values
        coefficients = dict(zip(feature_columns, sklearn_model.coef_))
        p_values = dict(zip(feature_columns, statsmodels_model.pvalues[1:]))  # Skip constant

        # Additional diagnostics
        r_squared = statsmodels_model.rsquared
        adjusted_r_squared = statsmodels_model.rsquared_adj
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)

        # Durbin-Watson test
        dw_stat = durbin_watson(statsmodels_model.resid)

        # Breusch-Pagan test
        try:
            bp_test = het_breuschpagan(statsmodels_model.resid, X_train_const)
            bp_pvalue = bp_test[1]
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            bp_pvalue = float('nan')

        # Feature importance (absolute coefficient values normalized)
        abs_coefs = np.abs(sklearn_model.coef_)
        feature_importance = dict(zip(
            feature_columns,
            abs_coefs / abs_coefs.sum() if abs_coefs.sum() > 0 else abs_coefs
        ))

        # Create model ID
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
        model_path = self.model_dir / f"{model_id}.joblib"

        # Package model artifacts
        model_artifacts = {
            "sklearn_model": sklearn_model,
            "statsmodels_model": statsmodels_model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "metrics": metrics,
            "coefficients": coefficients,
            "p_values": p_values,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "durbin_watson": dw_stat,
            "breusch_pagan_pvalue": bp_pvalue,
            "feature_importance": feature_importance,
            "training_timestamp": datetime.now(),
            "model_id": model_id
        }

        # Serialize with joblib
        async with self._lock:
            joblib.dump(model_artifacts, model_path)
            logger.info(f"Model saved to {model_path}")

        # Prepare response
        response = {
            "model_id": model_id,
            "training_timestamp": model_artifacts["training_timestamp"],
            "metrics": metrics,
            "coefficients": coefficients,
            "p_values": p_values,
            "r_squared": r_squared,
            "adjusted_r_squared": adjusted_r_squared,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "durbin_watson": dw_stat,
            "breusch_pagan_pvalue": bp_pvalue,
            "feature_importance": feature_importance,
            "model_path": str(model_path)
        }

        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        return response, str(model_path)

    async def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a serialized model."""
        try:
            async with self._lock:
                model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0"
)

model_manager = RegressionModelManager()


@app.post("/train", response_model=TrainingResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def train_model_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated list of feature columns")
):
    """
    Train a regression model from uploaded CSV data.
    
    Accepts multipart form data with:
    - file: CSV file
    - target_column: Target variable name
    - test_size: Test split ratio
    - random_state: Random seed
    - feature_columns: Optional comma-separated features
    """
    request_start = time.time()
    logger.info(f"Received training request for target={target_column}")

    # Validate request parameters
    try:
        request_data = TrainingRequest(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            feature_columns=feature_columns.split(",") if feature_columns else None
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Read and validate CSV
    try:
        content = await file.read()
        data = pd.read_csv(io.BytesIO(content))
        logger.info(f"Loaded CSV with shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if data.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Train model
    try:
        response, model_path = await model_manager.train_model(
            data=data,
            target_column=request_data.target_column,
            test_size=request_data.test_size,
            random_state=request_data.random_state,
            feature_columns=request_data.feature_columns
        )
        
        # Add processing time
        response["metrics"]["total_request_time"] = time.time() - request_start
        
        logger.info(f"Training successful: model_id={response['model_id']}")
        return response

    except ValueError as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_dir": str(model_manager.model_dir)
    }


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a trained model."""
    model_path = model_manager.model_dir / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        model = await model_manager.load_model(str(model_path))
        return {
            "model_id": model["model_id"],
            "training_timestamp": model["training_timestamp"],
            "metrics": model["metrics"],
            "r_squared": model["r_squared"],
            "adjusted_r_squared": model["adjusted_r_squared"],
            "feature_columns": model["feature_columns"],
            "target_column": model["target_column"]
        }
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for consistent error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            detail=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred",
            timestamp=datetime.now()
        ).dict()
    )


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5

# Phase 1: Core Model Training and Serialization - iteration 6
