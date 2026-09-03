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

# Type aliases
ModelArtifacts = Dict[str, Any]
TrainingResult = Dict[str, Any]


class ModelConfig(BaseModel):
    """Configuration for model training."""
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    fit_intercept: bool = Field(True, description="Whether to fit intercept")
    normalize: bool = Field(False, description="Whether to normalize features")


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of target variable")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    model_config: ModelConfig = Field(default_factory=ModelConfig)

    @validator("target_column")
    def validate_target_column(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()

    @validator("feature_columns")
    def validate_feature_columns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if len(v) == 0:
                raise ValueError("Feature columns cannot be empty")
            if len(set(v)) != len(v):
                raise ValueError("Feature columns must be unique")
            for col in v:
                if not col or not col.strip():
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
    timestamp: datetime
    model_path: str


class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._models: Dict[str, ModelArtifacts] = {}
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        config: Optional[ModelConfig] = None
    ) -> TrainingResult:
        """Train a regression model with full diagnostics."""
        start_time = time.time()
        config = config or ModelConfig()

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
                # Validate feature columns exist
                missing_cols = set(feature_columns) - set(data.columns)
                if missing_cols:
                    raise ValueError(f"Missing feature columns: {missing_cols}")

            # Extract and validate data
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
                raise ValueError("Insufficient data for training (minimum 10 samples)")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.test_size,
                random_state=config.random_state
            )

            # Train sklearn model
            sklearn_model = LinearRegression(
                fit_intercept=config.fit_intercept,
                normalize=config.normalize
            )
            sklearn_model.fit(X_train, y_train)

            # Predictions
            y_pred_train = sklearn_model.predict(X_train)
            y_pred_test = sklearn_model.predict(X_test)

            # Calculate metrics
            metrics = {
                "r2_train": float(r2_score(y_train, y_pred_train)),
                "r2_test": float(r2_score(y_test, y_pred_test)),
                "mse_train": float(mean_squared_error(y_train, y_pred_train)),
                "mse_test": float(mean_squared_error(y_test, y_pred_test)),
                "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
                "mae_test": float(mean_absolute_error(y_test, y_pred_test))
            }

            # Statsmodels for diagnostics
            X_with_const = add_constant(X_train)
            stats_model = OLS(y_train, X_with_const).fit()

            # Extract coefficients and p-values
            coefficients = {}
            p_values = {}
            for idx, col in enumerate(X_with_const.columns):
                if col == "const":
                    coefficients["intercept"] = float(stats_model.params[idx])
                    p_values["intercept"] = float(stats_model.pvalues[idx])
                else:
                    coefficients[col] = float(stats_model.params[idx])
                    p_values[col] = float(stats_model.pvalues[idx])

            # Additional diagnostics
            residuals = stats_model.resid
            dw_stat = float(durbin_watson(residuals))
            bp_test = het_breuschpagan(residuals, X_with_const)
            bp_stat, bp_pvalue, _, _ = bp_test

            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)

            # Generate model ID
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Prepare model artifacts
            model_artifacts = {
                "model_id": model_id,
                "sklearn_model": sklearn_model,
                "statsmodels_model": stats_model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "config": config.dict(),
                "metrics": metrics,
                "coefficients": coefficients,
                "p_values": p_values,
                "r_squared": float(stats_model.rsquared),
                "adjusted_r_squared": float(stats_model.rsquared_adj),
                "durbin_watson": dw_stat,
                "breusch_pagan_stat": bp_stat,
                "breusch_pagan_pvalue": bp_pvalue,
                "shapiro_stat": shapiro_stat,
                "shapiro_pvalue": shapiro_pvalue,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "timestamp": datetime.now(),
                "training_time": time.time() - start_time
            }

            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            await self._save_model(model_artifacts, model_path)
            model_artifacts["model_path"] = str(model_path)

            # Store in memory
            async with self._lock:
                self._models[model_id] = model_artifacts

            logger.info(
                f"Model trained successfully: {model_id}, "
                f"R²={metrics['r2_test']:.4f}, "
                f"training_time={model_artifacts['training_time']:.2f}s"
            )

            return model_artifacts

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    async def _save_model(self, model_artifacts: ModelArtifacts, path: Path) -> None:
        """Asynchronously save model artifacts to disk."""
        try:
            # Run joblib save in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                joblib.dump,
                model_artifacts,
                path
            )
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    async def load_model(self, model_id: str) -> ModelArtifacts:
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
            model_artifacts = await asyncio.get_event_loop().run_in_executor(
                None,
                joblib.load,
                model_path
            )
            async with self._lock:
                self._models[model_id] = model_artifacts
            return model_artifacts
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    async def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        model_artifacts = await self.load_model(model_id)
        sklearn_model = model_artifacts["sklearn_model"]
        feature_columns = model_artifacts["feature_columns"]

        # Validate input data
        missing_cols = set(feature_columns) - set(data.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_cols}"
            )

        X = data[feature_columns]
        try:
            predictions = sklearn_model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with basic info."""
        models_info = []
        for model_id, artifacts in self._models.items():
            models_info.append({
                "model_id": model_id,
                "r_squared": artifacts.get("r_squared"),
                "timestamp": artifacts.get("timestamp"),
                "training_samples": artifacts.get("training_samples"),
                "test_samples": artifacts.get("test_samples")
            })
        return models_info


# FastAPI application
app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0"
)

# Global model manager instance
model_manager = ModelManager()


@app.post("/train", response_model=TrainingResponse)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of target column"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns"),
    test_size: float = File(0.2, ge=0.1, le=0.5),
    random_state: int = File(42, ge=0)
):
    """Train a regression model from uploaded CSV data."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read CSV data
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        # Parse feature columns if provided
        feature_list = None
        if feature_columns:
            feature_list = [col.strip() for col in feature_columns.split(",") if col.strip()]

        # Create config
        config = ModelConfig(
            test_size=test_size,
            random_state=random_state
        )

        # Train model
        result = await model_manager.train_model(
            data=data,
            target_column=target_column,
            feature_columns=feature_list,
            config=config
        )

        # Prepare response
        response = TrainingResponse(
            model_id=result["model_id"],
            metrics=result["metrics"],
            coefficients=result["coefficients"],
            p_values=result["p_values"],
            r_squared=result["r_squared"],
            adjusted_r_squared=result["adjusted_r_squared"],
            mse=result["metrics"]["mse_test"],
            rmse=result["metrics"]["rmse_test"],
            mae=result["metrics"]["mae_test"],
            training_samples=result["training_samples"],
            test_samples=result["test_samples"],
            timestamp=result["timestamp"],
            model_path=result["model_path"]
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/models")
async def list_models_endpoint():
    """List all trained models."""
    try:
        models = await model_manager.list_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        logger.error(f"List models error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_endpoint(model_id: str):
    """Get model details and diagnostics."""
    try:
        model = await model_manager.load_model(model_id)
        return {
            "model_id": model["model_id"],
            "metrics": model["metrics"],
            "coefficients": model["coefficients"],
            "p_values": model["p_values"],
            "r_squared": model["r_squared"],
            "adjusted_r_squared": model["adjusted_r_squared"],
            "durbin_watson": model["durbin_watson"],
            "breusch_pagan_stat": model["breusch_pagan_stat"],
            "breusch_pagan_pvalue": model["breusch_pagan_pvalue"],
            "shapiro_stat": model["shapiro_stat"],
            "shapiro_pvalue": model["shapiro_pvalue"],
            "training_samples": model["training_samples"],
            "test_samples": model["test_samples"],
            "timestamp": model["timestamp"],
            "training_time": model["training_time"],
            "model_path": model["model_path"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get model error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")


@app.post("/predict/{model_id}")
async def predict_endpoint(
    model_id: str,
    file: UploadFile = File(..., description="CSV file with features for prediction")
):
    """Make predictions using a trained model."""
    try:
        # Read CSV data
        content = await file.read()
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

        # Make predictions
        predictions = await model_manager.predict(model_id, data)

        return {
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_manager._models)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3
