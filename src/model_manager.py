import asyncio
import io
import logging
import os
import tempfile
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
logger = logging.getLogger("regression_api")

# Constants
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
SUPPORTED_EXTENSIONS = {".csv", ".txt"}


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns. If None, all other columns used")
    
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
            cleaned = [col.strip() for col in v if col.strip()]
            if len(cleaned) != len(v):
                raise ValueError("Feature columns cannot contain empty strings")
            return cleaned
        return v


class TrainingResponse(BaseModel):
    """Response model for training results."""
    model_id: str
    metrics: Dict[str, Any]
    diagnostics: Dict[str, Any]
    feature_importance: Dict[str, float]
    training_timestamp: str
    model_path: str


class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with directory: {self.model_dir}")
    
    async def train_and_save(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_state: int,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train OLS model, compute diagnostics, and save model."""
        
        async with self._lock:
            try:
                logger.info(f"Starting training with target={target_column}, test_size={test_size}")
                
                # Validate data
                if data.empty:
                    raise ValueError("Data is empty")
                
                if target_column not in data.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                
                # Prepare features
                if feature_columns is None:
                    feature_columns = [col for col in data.columns if col != target_column]
                else:
                    missing_cols = set(feature_columns) - set(data.columns)
                    if missing_cols:
                        raise ValueError(f"Missing feature columns: {missing_cols}")
                
                if not feature_columns:
                    raise ValueError("No feature columns available for training")
                
                # Extract data
                X = data[feature_columns].copy()
                y = data[target_column].copy()
                
                # Check for non-numeric data
                if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                    raise ValueError("All feature columns must be numeric")
                if not pd.api.types.is_numeric_dtype(y):
                    raise ValueError("Target column must be numeric")
                
                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    logger.warning("Missing values detected, dropping rows")
                    mask = X.notnull().all(axis=1) & y.notnull()
                    X = X[mask]
                    y = y[mask]
                
                if len(X) < 10:
                    raise ValueError("Insufficient data after cleaning (minimum 10 rows required)")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train sklearn model
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = sklearn_model.predict(X_train)
                y_pred_test = sklearn_model.predict(X_test)
                
                # Compute metrics
                metrics = {
                    "r2_train": float(r2_score(y_train, y_pred_train)),
                    "r2_test": float(r2_score(y_test, y_pred_test)),
                    "mse_train": float(mean_squared_error(y_train, y_pred_train)),
                    "mse_test": float(mean_squared_error(y_test, y_pred_test)),
                    "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
                    "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
                    "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                    "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                    "n_samples": int(len(X)),
                    "n_features": int(X.shape[1]),
                    "n_train_samples": int(len(X_train)),
                    "n_test_samples": int(len(X_test))
                }
                
                # Statsmodels diagnostics
                X_with_const = add_constant(X)
                statsmodels_model = OLS(y, X_with_const).fit()
                
                # Extract coefficients and p-values
                coefficients = {}
                p_values = {}
                for i, col in enumerate(feature_columns):
                    coefficients[col] = float(sklearn_model.coef_[i])
                    p_values[col] = float(statsmodels_model.pvalues[i + 1])  # +1 for constant
                
                # Intercept
                coefficients["intercept"] = float(sklearn_model.intercept_)
                p_values["intercept"] = float(statsmodels_model.pvalues[0])
                
                # Additional diagnostics
                residuals = y - sklearn_model.predict(X)
                
                # Durbin-Watson statistic
                dw_stat = durbin_watson(residuals)
                
                # Breusch-Pagan test for heteroscedasticity
                try:
                    bp_test = het_breuschpagan(residuals, X_with_const)
                    bp_stat, bp_pvalue, bp_fvalue, bp_fpvalue = bp_test
                except Exception as e:
                    logger.warning(f"Breusch-Pagan test failed: {e}")
                    bp_stat, bp_pvalue, bp_fvalue, bp_fpvalue = None, None, None, None
                
                # Shapiro-Wilk test for normality of residuals
                try:
                    shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
                except Exception as e:
                    logger.warning(f"Shapiro-Wilk test failed: {e}")
                    shapiro_stat, shapiro_pvalue = None, None
                
                diagnostics = {
                    "coefficients": coefficients,
                    "p_values": p_values,
                    "standard_errors": {col: float(statsmodels_model.bse[i + 1]) for i, col in enumerate(feature_columns)},
                    "confidence_intervals": {
                        col: statsmodels_model.conf_int()[i + 1].tolist() 
                        for i, col in enumerate(feature_columns)
                    },
                    "aic": float(statsmodels_model.aic),
                    "bic": float(statsmodels_model.bic),
                    "f_statistic": float(statsmodels_model.fvalue),
                    "f_pvalue": float(statsmodels_model.f_pvalue),
                    "durbin_watson": float(dw_stat),
                    "breusch_pagan": {
                        "statistic": float(bp_stat) if bp_stat is not None else None,
                        "pvalue": float(bp_pvalue) if bp_pvalue is not None else None
                    },
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat) if shapiro_stat is not None else None,
                        "pvalue": float(shapiro_pvalue) if shapiro_pvalue is not None else None
                    },
                    "residual_mean": float(np.mean(residuals)),
                    "residual_std": float(np.std(residuals)),
                    "adjusted_r2": float(statsmodels_model.rsquared_adj)
                }
                
                # Feature importance (absolute coefficient values)
                feature_importance = {
                    col: abs(float(coef)) for col, coef in zip(feature_columns, sklearn_model.coef_)
                }
                # Normalize to percentages
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    feature_importance = {
                        col: (val / total_importance) * 100 
                        for col, val in feature_importance.items()
                    }
                
                # Generate model ID
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
                model_path = self.model_dir / f"{model_id}.joblib"
                
                # Package model artifacts
                model_artifacts = {
                    "sklearn_model": sklearn_model,
                    "statsmodels_model": statsmodels_model,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "model_metadata": {
                        "model_id": model_id,
                        "training_date": datetime.now().isoformat(),
                        "metrics": metrics,
                        "diagnostics": diagnostics,
                        "feature_importance": feature_importance
                    }
                }
                
                # Save model
                await asyncio.to_thread(joblib.dump, model_artifacts, model_path)
                logger.info(f"Model saved to {model_path}")
                
                return {
                    "model_id": model_id,
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "feature_importance": feature_importance,
                    "training_timestamp": datetime.now().isoformat(),
                    "model_path": str(model_path)
                }
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


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
async def train_model(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
) -> Dict[str, Any]:
    """
    Train a regression model from uploaded CSV data.
    
    Args:
        file: CSV file upload
        target_column: Name of the target column
        test_size: Proportion of data for testing (0.1-0.5)
        random_state: Random seed for reproducibility
        feature_columns: Optional comma-separated list of feature columns
    
    Returns:
        TrainingResponse with model metrics and diagnostics
    """
    try:
        # Validate file extension
        filename = file.filename or ""
        file_ext = Path(filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
            )
        
        # Read file content with size limit
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)} MB"
            )
        
        # Parse feature columns
        feature_list = None
        if feature_columns:
            feature_list = [col.strip() for col in feature_columns.split(",") if col.strip()]
        
        # Validate request parameters
        try:
            request = TrainingRequest(
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                feature_columns=feature_list
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        # Parse CSV data
        try:
            data = pd.read_csv(io.BytesIO(content))
            logger.info(f"Loaded CSV with {len(data)} rows and {len(data.columns)} columns")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        # Train model
        result = await model_manager.train_and_save(
            data=data,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            feature_columns=request.feature_columns
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    try:
        model_path = MODEL_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model_artifacts = await asyncio.to_thread(joblib.load, model_path)
        metadata = model_artifacts.get("model_metadata", {})
        
        return {
            "model_id": model_id,
            "metadata": metadata,
            "model_path": str(model_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all trained models."""
    try:
        model_files = list(MODEL_DIR.glob("*.joblib"))
        model_ids = [f.stem for f in model_files]
        return {"models": model_ids, "count": len(model_ids)}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4
