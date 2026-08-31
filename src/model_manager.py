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

# Configure logging
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
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
SUPPORTED_EXTENSIONS = {".csv", ".txt"}


class TrainingData(BaseModel):
    """Pydantic model for training data validation."""
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


class ModelMetrics(BaseModel):
    """Pydantic model for model metrics."""
    r2_score: float
    adjusted_r2: float
    mse: float
    rmse: float
    mae: float
    coefficients: Dict[str, float]
    intercept: float
    p_values: Dict[str, float]
    f_statistic: float
    f_p_value: float
    aic: float
    bic: float
    durbin_watson: float
    breusch_pagan_p_value: float
    training_samples: int
    test_samples: int
    feature_count: int
    training_time: float
    model_version: str
    created_at: str


class ModelManager:
    """Manages regression model training, evaluation, and serialization."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Any, ModelMetrics]:
        """Train a regression model with comprehensive diagnostics."""
        start_time = time.time()
        logger.info(f"Starting model training with target: {target_column}")

        try:
            # Validate data
            if data.empty:
                raise ValueError("Training data is empty")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            if data[target_column].isnull().any():
                raise ValueError("Target column contains null values")

            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            else:
                missing_cols = set(feature_columns) - set(data.columns)
                if missing_cols:
                    raise ValueError(f"Missing feature columns: {missing_cols}")

            if not feature_columns:
                raise ValueError("No feature columns available for training")

            # Prepare data
            X = data[feature_columns].copy()
            y = data[target_column].copy()

            # Validate numeric data
            for col in feature_columns:
                if not np.issubdtype(X[col].dtype, np.number):
                    raise ValueError(f"Feature column '{col}' must be numeric")
                if X[col].isnull().any():
                    raise ValueError(f"Feature column '{col}' contains null values")

            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("Target column must be numeric")

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

            # Calculate metrics
            r2 = r2_score(y_test, y_pred_test)
            n = len(X_test)
            p = len(feature_columns)
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred_test)

            # Extract coefficients and p-values
            coefficients = dict(zip(feature_columns, sklearn_model.coef_))
            p_values = dict(zip(feature_columns, stats_model.pvalues[1:]))

            # Additional diagnostics
            f_statistic = stats_model.fvalue
            f_p_value = stats_model.f_pvalue
            aic = stats_model.aic
            bic = stats_model.bic
            dw_statistic = durbin_watson(stats_model.resid)

            # Breusch-Pagan test for heteroscedasticity
            try:
                bp_test = het_breuschpagan(stats_model.resid, X_train_const)
                bp_p_value = bp_test[1]
            except Exception as e:
                logger.warning(f"Breusch-Pagan test failed: {e}")
                bp_p_value = float("nan")

            # Create metrics
            metrics = ModelMetrics(
                r2_score=float(r2),
                adjusted_r2=float(adjusted_r2),
                mse=float(mse),
                rmse=float(rmse),
                mae=float(mae),
                coefficients={k: float(v) for k, v in coefficients.items()},
                intercept=float(sklearn_model.intercept_),
                p_values={k: float(v) for k, v in p_values.items()},
                f_statistic=float(f_statistic),
                f_p_value=float(f_p_value),
                aic=float(aic),
                bic=float(bic),
                durbin_watson=float(dw_statistic),
                breusch_pagan_p_value=float(bp_p_value),
                training_samples=len(X_train),
                test_samples=len(X_test),
                feature_count=len(feature_columns),
                training_time=time.time() - start_time,
                model_version="1.0.0",
                created_at=datetime.utcnow().isoformat()
            )

            # Store model with metadata
            model_data = {
                "sklearn_model": sklearn_model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "metrics": metrics.dict(),
                "training_date": datetime.utcnow().isoformat()
            }

            logger.info(f"Model training completed in {metrics.training_time:.2f} seconds")
            logger.info(f"R² score: {r2:.4f}, RMSE: {rmse:.4f}")

            return model_data, metrics

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    async def save_model(self, model_data: Dict[str, Any], model_name: str) -> Path:
        """Save model to disk with joblib."""
        async with self._lock:
            try:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"{model_name}_{timestamp}.joblib"
                filepath = self.model_dir / filename

                # Save model
                await asyncio.to_thread(joblib.dump, model_data, filepath)
                logger.info(f"Model saved to {filepath}")

                # Save metadata separately
                metadata_path = self.model_dir / f"{model_name}_{timestamp}_metadata.json"
                import json
                with open(metadata_path, "w") as f:
                    json.dump(model_data["metrics"], f, indent=2)

                return filepath

            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")
                raise

    async def load_model(self, model_path: Path) -> Dict[str, Any]:
        """Load model from disk."""
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model_data = await asyncio.to_thread(joblib.load, model_path)
            logger.info(f"Model loaded from {model_path}")
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise


# FastAPI app
app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0"
)

model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/train", response_model=ModelMetrics)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
) -> ModelMetrics:
    """Train a regression model from uploaded CSV data."""
    logger.info(f"Received training request: target={target_column}, test_size={test_size}")

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
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
            )

        # Parse CSV
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

        # Validate training parameters
        try:
            training_config = TrainingData(
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                feature_columns=feature_columns.split(",") if feature_columns else None
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Train model
        try:
            model_data, metrics = await model_manager.train_model(
                data=data,
                target_column=training_config.target_column,
                feature_columns=training_config.feature_columns,
                test_size=training_config.test_size,
                random_state=training_config.random_state
            )

            # Save model
            model_name = f"regression_{training_config.target_column}"
            await model_manager.save_model(model_data, model_name)

            return metrics

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all saved models."""
    try:
        models = [f.name for f in model_manager.model_dir.glob("*.joblib")]
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.get("/models/{model_name}")
async def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    try:
        model_files = list(model_manager.model_dir.glob(f"{model_name}_*.joblib"))
        if not model_files:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        # Get the most recent model
        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        model_data = await model_manager.load_model(latest_model)

        return {
            "model_name": model_name,
            "file_path": str(latest_model),
            "metrics": model_data["metrics"],
            "feature_columns": model_data["feature_columns"],
            "target_column": model_data["target_column"],
            "training_date": model_data["training_date"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model info")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5
