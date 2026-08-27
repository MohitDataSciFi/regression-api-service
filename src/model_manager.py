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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("regression_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== Schemas ====================
class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns (default: all except target)")

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
    """Response model for training endpoint."""
    model_id: str
    training_timestamp: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    mse: float
    rmse: float
    mae: float
    feature_importance: Dict[str, float]
    diagnostics: Dict[str, Any]
    model_path: str


class PredictionRequest(BaseModel):
    """Pydantic model for prediction request."""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: float
    model_id: str
    timestamp: str


# ==================== Model Manager ====================
class ModelManager:
    """Manages regression model training, serialization, and diagnostics."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

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
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        logger.info(f"Starting model training with ID: {model_id}")

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

            # Extract data
            X = data[feature_columns].copy()
            y = data[target_column].copy()

            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning("Missing values detected, dropping rows with NaN")
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]

            # Convert to numeric
            X = X.apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')

            # Drop rows with NaN after conversion
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

            if len(X) == 0:
                raise ValueError("No valid data after cleaning")

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

            # Make predictions
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
                "test_mae": mean_absolute_error(y_test, y_pred_test),
            }

            # Coefficients and p-values
            coefficients = dict(zip(feature_columns, sklearn_model.coef_))
            p_values = dict(zip(feature_columns, statsmodels_model.pvalues[1:]))  # Skip constant

            # Feature importance (absolute coefficients normalized)
            abs_coefs = np.abs(sklearn_model.coef_)
            feature_importance = dict(zip(
                feature_columns,
                abs_coefs / abs_coefs.sum() if abs_coefs.sum() > 0 else abs_coefs
            ))

            # Diagnostics
            diagnostics = {
                "durbin_watson": float(durbin_watson(statsmodels_model.resid)),
                "condition_number": float(np.linalg.cond(X_train_const)),
                "aic": float(statsmodels_model.aic),
                "bic": float(statsmodels_model.bic),
                "f_statistic": float(statsmodels_model.fvalue),
                "f_p_value": float(statsmodels_model.f_pvalue),
                "nobs": int(statsmodels_model.nobs),
                "df_model": int(statsmodels_model.df_model),
                "df_resid": int(statsmodels_model.df_resid),
            }

            # Breusch-Pagan test for heteroscedasticity
            try:
                bp_test = het_breuschpagan(statsmodels_model.resid, X_train_const)
                diagnostics["breusch_pagan_lm"] = float(bp_test[0])
                diagnostics["breusch_pagan_pvalue"] = float(bp_test[1])
            except Exception as e:
                logger.warning(f"Breusch-Pagan test failed: {e}")
                diagnostics["breusch_pagan_lm"] = None
                diagnostics["breusch_pagan_pvalue"] = None

            # Prepare model artifact
            model_artifact = {
                "sklearn_model": sklearn_model,
                "statsmodels_model": statsmodels_model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "model_id": model_id,
                "training_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "test_size": test_size,
                    "random_state": random_state,
                    "n_samples": len(X),
                    "n_features": len(feature_columns),
                    "training_time": time.time() - start_time
                }
            }

            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            await self._save_model(model_artifact, model_path)

            # Store in memory
            async with self._lock:
                self._models[model_id] = {
                    "model": model_artifact,
                    "path": str(model_path),
                    "metrics": metrics,
                    "coefficients": coefficients,
                    "p_values": p_values,
                    "feature_importance": feature_importance,
                    "diagnostics": diagnostics
                }

            logger.info(f"Model {model_id} trained successfully in {time.time() - start_time:.2f}s")
            
            return {
                "model_id": model_id,
                "training_timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "coefficients": coefficients,
                "p_values": p_values,
                "r_squared": metrics["test_r2"],
                "adjusted_r_squared": float(statsmodels_model.rsquared_adj),
                "mse": metrics["test_mse"],
                "rmse": metrics["test_rmse"],
                "mae": metrics["test_mae"],
                "feature_importance": feature_importance,
                "diagnostics": diagnostics,
                "model_path": str(model_path)
            }

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    async def predict(self, model_id: str, features: Dict[str, float]) -> float:
        """Make prediction using a trained model."""
        async with self._lock:
            if model_id not in self._models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            model_info = self._models[model_id]
            model = model_info["model"]["sklearn_model"]
            feature_columns = model_info["model"]["feature_columns"]

        # Validate features
        missing_features = set(feature_columns) - set(features.keys())
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

        # Prepare feature vector
        feature_vector = np.array([[features[col] for col in feature_columns]])
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        
        return float(prediction)

    async def _save_model(self, model_artifact: Dict[str, Any], path: Path) -> None:
        """Save model artifact to disk."""
        try:
            # Run in thread pool to avoid blocking
            await asyncio.to_thread(joblib.dump, model_artifact, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load model from disk if not in memory."""
        async with self._lock:
            if model_id in self._models:
                return self._models[model_id]
        
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        try:
            model_artifact = await asyncio.to_thread(joblib.load, model_path)
            async with self._lock:
                self._models[model_id] = {
                    "model": model_artifact,
                    "path": str(model_path)
                }
            return self._models[model_id]
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


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
async def train_model(
    file: UploadFile = File(..., description="CSV file with training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
):
    """Train a regression model from CSV data."""
    try:
        # Validate request parameters
        request_data = {
            "target_column": target_column,
            "test_size": test_size,
            "random_state": random_state,
            "feature_columns": feature_columns.split(",") if feature_columns else None
        }
        
        try:
            training_request = TrainingRequest(**request_data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

        # Read and validate CSV
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

        # Train model
        result = await model_manager.train_model(
            data=data,
            target_column=training_request.target_column,
            test_size=training_request.test_size,
            random_state=training_request.random_state,
            feature_columns=training_request.feature_columns
        )

        return TrainingResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict(model_id: str, request: PredictionRequest):
    """Make prediction using a trained model."""
    try:
        prediction = await model_manager.predict(model_id, request.features)
        return PredictionResponse(
            prediction=prediction,
            model_id=model_id,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all available models."""
    model_files = list(model_manager.model_dir.glob("*.joblib"))
    model_ids = [f.stem for f in model_files]
    return {"models": model_ids}


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get model information and metrics."""
    try:
        model_info = await model_manager.load_model(model_id)
        return {
            "model_id": model_id,
            "path": model_info.get("path"),
            "metrics": model_info.get("metrics", {}),
            "coefficients": model_info.get("coefficients", {}),
            "p_values": model_info.get("p_values", {}),
            "feature_importance": model_info.get("feature_importance", {}),
            "diagnostics": model_info.get("diagnostics", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.delete("/models/{model_id}")
async def delete_model(model_id: str) -> Dict[str, str]:
    """Delete a trained model."""
    try:
        model_path = model_manager.model_dir / f"{model_id}.joblib"
        if model_path.exists():
            await asyncio.to_thread(model_path.unlink)
            async with model_manager._lock:
                model_manager._models.pop(model_id, None)
            logger.info(f"Model {model_id} deleted")
            return {"status": "deleted", "model_id": model_id}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4
