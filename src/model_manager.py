import asyncio
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
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("regression_api.log")
    ]
)
logger = logging.getLogger(__name__)


# ==================== Schemas ====================
class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test split ratio")
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


class ModelMetrics(BaseModel):
    """Metrics returned after model training."""
    r2_train: float
    r2_test: float
    mse_train: float
    mse_test: float
    mae_train: float
    mae_test: float
    coefficients: Dict[str, float]
    intercept: float
    p_values: Dict[str, float]
    f_statistic: float
    f_p_value: float
    durbin_watson: float
    breusch_pagan_p_value: float
    training_samples: int
    test_samples: int
    feature_count: int
    model_path: str
    trained_at: datetime


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""
    success: bool
    metrics: Optional[ModelMetrics] = None
    error: Optional[str] = None


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
    ) -> ModelMetrics:
        """Train an OLS regression model with diagnostics."""
        async with self._lock:
            try:
                logger.info(f"Starting model training with target: {target_column}")
                
                # Validate data
                if data.empty:
                    raise ValueError("Data is empty")
                
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
                
                if not feature_columns:
                    raise ValueError("No feature columns available for training")
                
                # Extract data
                X = data[feature_columns].copy()
                y = data[target_column].copy()
                
                # Check for non-numeric data
                if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                    raise ValueError("All feature columns must be numeric")
                if not np.issubdtype(y.dtype, np.number):
                    raise ValueError("Target column must be numeric")
                
                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    logger.warning("Missing values detected, dropping rows")
                    mask = X.notnull().all(axis=1) & y.notnull()
                    X = X[mask]
                    y = y[mask]
                
                if len(X) < 10:
                    raise ValueError("Insufficient data for training (minimum 10 samples)")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
                
                # Train scikit-learn model
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)
                
                # Train statsmodels for diagnostics
                X_train_const = add_constant(X_train)
                stats_model = OLS(y_train, X_train_const).fit()
                
                # Compute predictions
                y_train_pred = sklearn_model.predict(X_train)
                y_test_pred = sklearn_model.predict(X_test)
                
                # Compute metrics
                r2_train = r2_score(y_train, y_train_pred)
                r2_test = r2_score(y_test, y_test_pred)
                mse_train = mean_squared_error(y_train, y_train_pred)
                mse_test = mean_squared_error(y_test, y_test_pred)
                mae_train = mean_absolute_error(y_train, y_train_pred)
                mae_test = mean_absolute_error(y_test, y_test_pred)
                
                # Extract coefficients and p-values
                coefficients = dict(zip(feature_columns, sklearn_model.coef_))
                p_values = {col: float(stats_model.pvalues[col]) for col in feature_columns}
                
                # Diagnostic tests
                residuals = y_test - y_test_pred
                dw_stat = durbin_watson(residuals)
                
                # Breusch-Pagan test
                X_test_const = add_constant(X_test)
                try:
                    bp_test = het_breuschpagan(residuals, X_test_const)
                    bp_p_value = float(bp_test[1])
                except Exception as e:
                    logger.warning(f"Breusch-Pagan test failed: {e}")
                    bp_p_value = float("nan")
                
                # Serialize model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"regression_model_{timestamp}.joblib"
                model_path = self.model_dir / model_filename
                
                model_data = {
                    "model": sklearn_model,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "metrics": {
                        "r2_train": r2_train,
                        "r2_test": r2_test,
                        "mse_train": mse_train,
                        "mse_test": mse_test,
                        "mae_train": mae_train,
                        "mae_test": mae_test
                    },
                    "trained_at": datetime.now().isoformat()
                }
                
                # Save model asynchronously
                await asyncio.to_thread(joblib.dump, model_data, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Build metrics response
                metrics = ModelMetrics(
                    r2_train=float(r2_train),
                    r2_test=float(r2_test),
                    mse_train=float(mse_train),
                    mse_test=float(mse_test),
                    mae_train=float(mae_train),
                    mae_test=float(mae_test),
                    coefficients=coefficients,
                    intercept=float(sklearn_model.intercept_),
                    p_values=p_values,
                    f_statistic=float(stats_model.fvalue),
                    f_p_value=float(stats_model.f_pvalue),
                    durbin_watson=float(dw_stat),
                    breusch_pagan_p_value=bp_p_value,
                    training_samples=len(X_train),
                    test_samples=len(X_test),
                    feature_count=len(feature_columns),
                    model_path=str(model_path),
                    trained_at=datetime.now()
                )
                
                logger.info("Model training completed successfully")
                return metrics
                
            except Exception as e:
                logger.error(f"Model training failed: {str(e)}", exc_info=True)
                raise


# ==================== API ====================
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
async def train_model_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test split ratio (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
) -> TrainingResponse:
    """Train a regression model from uploaded CSV data."""
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
            logger.warning(f"Validation error: {e}")
            return TrainingResponse(success=False, error=str(e))
        
        # Read and validate CSV file
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Parse CSV
        try:
            data = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            return TrainingResponse(success=False, error=f"Invalid CSV format: {str(e)}")
        
        # Train model
        try:
            metrics = await model_manager.train_model(
                data=data,
                target_column=request.target_column,
                test_size=request.test_size,
                random_state=request.random_state,
                feature_columns=request.feature_columns
            )
            return TrainingResponse(success=True, metrics=metrics)
            
        except ValueError as e:
            logger.error(f"Training error: {e}")
            return TrainingResponse(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected training error: {e}", exc_info=True)
            return TrainingResponse(success=False, error=f"Training failed: {str(e)}")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected endpoint error: {e}", exc_info=True)
        return TrainingResponse(success=False, error=f"Unexpected error: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all trained models."""
    try:
        models = [f.name for f in model_manager.model_dir.glob("*.joblib")]
        return {"models": sorted(models)}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4
