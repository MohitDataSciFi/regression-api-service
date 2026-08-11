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
import statsmodels.api as sm
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
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
        logging.FileHandler("regression_api.log")
    ]
)
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for model training."""
    target_column: str = Field(..., description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns. If None, all other columns used")

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()


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
        config: ModelConfig
    ) -> Tuple[Dict[str, Any], str]:
        """Train a regression model with diagnostics and return metrics and model path."""
        
        async with self._lock:
            try:
                logger.info(f"Starting model training with config: {config}")
                
                # Validate data
                if data.empty:
                    raise ValueError("Data is empty")
                
                if config.target_column not in data.columns:
                    raise ValueError(f"Target column '{config.target_column}' not found in data")
                
                # Prepare features
                if config.feature_columns:
                    feature_cols = [col for col in config.feature_columns if col in data.columns]
                    if len(feature_cols) != len(config.feature_columns):
                        missing = set(config.feature_columns) - set(feature_cols)
                        raise ValueError(f"Missing feature columns: {missing}")
                else:
                    feature_cols = [col for col in data.columns if col != config.target_column]
                
                if not feature_cols:
                    raise ValueError("No feature columns available for training")
                
                # Extract data
                X = data[feature_cols].copy()
                y = data[config.target_column].copy()
                
                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    logger.warning("Missing values detected, dropping rows")
                    mask = X.notnull().all(axis=1) & y.notnull()
                    X = X[mask]
                    y = y[mask]
                
                # Convert to numeric
                X = X.apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(y, errors='coerce')
                
                # Drop any remaining NaN rows
                mask = X.notnull().all(axis=1) & y.notnull()
                X = X[mask]
                y = y[mask]
                
                if len(X) < 10:
                    raise ValueError("Insufficient data after cleaning (minimum 10 rows required)")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=config.test_size, 
                    random_state=config.random_state
                )
                
                # Train scikit-learn model
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)
                
                # Train statsmodels for diagnostics
                X_train_sm = sm.add_constant(X_train)
                sm_model = sm.OLS(y_train, X_train_sm).fit()
                
                # Compute predictions
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
                    "n_features": len(feature_cols),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test)
                }
                
                # Extract coefficients and p-values
                coefficients = {}
                p_values = {}
                feature_importance = {}
                
                for i, col in enumerate(feature_cols):
                    coef = float(sklearn_model.coef_[i])
                    coefficients[col] = coef
                    feature_importance[col] = abs(coef)
                    
                    # Get p-value from statsmodels
                    if col in sm_model.params.index:
                        p_values[col] = float(sm_model.pvalues[col])
                    else:
                        p_values[col] = float('nan')
                
                # Normalize feature importance
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
                
                # Create model ID
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)}"
                
                # Prepare model artifact
                model_artifact = {
                    "sklearn_model": sklearn_model,
                    "statsmodels_model": sm_model,
                    "feature_columns": feature_cols,
                    "target_column": config.target_column,
                    "config": config.model_dump(),
                    "metrics": metrics,
                    "coefficients": coefficients,
                    "p_values": p_values,
                    "feature_importance": feature_importance,
                    "training_timestamp": datetime.now().isoformat(),
                    "model_id": model_id
                }
                
                # Serialize model
                model_path = self.model_dir / f"{model_id}.joblib"
                await asyncio.to_thread(joblib.dump, model_artifact, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Prepare response
                response = {
                    "model_id": model_id,
                    "training_timestamp": model_artifact["training_timestamp"],
                    "metrics": metrics,
                    "coefficients": coefficients,
                    "p_values": p_values,
                    "r_squared": float(sm_model.rsquared),
                    "adjusted_r_squared": float(sm_model.rsquared_adj),
                    "feature_importance": feature_importance,
                    "model_path": str(model_path),
                    "data_shape": {"rows": len(X), "columns": len(feature_cols) + 1}
                }
                
                logger.info(f"Model training completed successfully. Model ID: {model_id}")
                return response, model_id
                
            except Exception as e:
                logger.error(f"Model training failed: {str(e)}", exc_info=True)
                raise


class TrainingRequest(BaseModel):
    """Request model for training endpoint."""
    config: ModelConfig


# Initialize FastAPI app
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
) -> TrainingResponse:
    """Train a regression model from uploaded CSV data."""
    
    try:
        logger.info(f"Received training request: file={file.filename}, target={target_column}")
        
        # Validate file extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read and parse CSV
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to parse CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        
        # Parse feature columns
        feature_list = None
        if feature_columns:
            feature_list = [col.strip() for col in feature_columns.split(',') if col.strip()]
        
        # Create config
        config = ModelConfig(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            feature_columns=feature_list
        )
        
        # Train model
        try:
            response, model_id = await model_manager.train_model(data, config)
            return TrainingResponse(**response)
        except ValueError as e:
            logger.error(f"Training validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    try:
        model_path = model_manager.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model_artifact = await asyncio.to_thread(joblib.load, model_path)
        
        return {
            "model_id": model_artifact["model_id"],
            "training_timestamp": model_artifact["training_timestamp"],
            "metrics": model_artifact["metrics"],
            "coefficients": model_artifact["coefficients"],
            "p_values": model_artifact["p_values"],
            "r_squared": model_artifact["metrics"].get("test_r2"),
            "feature_columns": model_artifact["feature_columns"],
            "target_column": model_artifact["target_column"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving model: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all trained models."""
    try:
        model_files = list(model_manager.model_dir.glob("*.joblib"))
        model_ids = [f.stem for f in model_files]
        return {"models": model_ids}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5
