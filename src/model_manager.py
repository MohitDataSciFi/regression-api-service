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
        logging.StreamHandler(),
        logging.FileHandler("regression_api.log")
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
    feature_columns: Optional[List[str]] = Field(
        None, description="List of feature columns. If None, all other columns are used."
    )

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
            if len(set(v)) != len(v):
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
    training_time: float
    created_at: datetime
    model_path: str


class ModelManager:
    """Manages model training, serialization, and diagnostics."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> TrainingResponse:
        """Train an OLS regression model with diagnostics."""
        start_time = time.time()
        logger.info(f"Starting model training with target: {target_column}")

        # Validate data
        if data.empty:
            raise ValueError("Training data is empty")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Prepare features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        else:
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

        # Validate all columns exist
        all_columns = feature_columns + [target_column]
        missing_columns = set(all_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")

        # Extract features and target
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Check for non-numeric data
        if not np.issubdtype(X.dtypes.values[0], np.number) if len(X.columns) > 0 else False:
            raise ValueError("Feature columns must be numeric")
        
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Target column must be numeric")

        # Handle missing values
        if X.isnull().any().any() or y.isnull().any():
            logger.warning("Missing values detected, dropping rows")
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

        if len(X) == 0:
            raise ValueError("No valid data after cleaning")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Train sklearn model
        sklearn_model = LinearRegression()
        sklearn_model.fit(X_train, y_train)

        # Train statsmodels for diagnostics
        X_train_const = add_constant(X_train)
        statsmodels_model = OLS(y_train, X_train_const).fit()

        # Compute predictions and metrics
        y_pred_train = sklearn_model.predict(X_train)
        y_pred_test = sklearn_model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)

        # Diagnostics from statsmodels
        coefficients = statsmodels_model.params[1:].to_dict()  # Exclude constant
        p_values = statsmodels_model.pvalues[1:].to_dict()  # Exclude constant
        adjusted_r2 = statsmodels_model.rsquared_adj

        # Additional diagnostics
        residuals = y_test - y_pred_test
        dw_stat = durbin_watson(residuals)
        bp_test = het_breuschpagan(residuals, X_test)

        # Create model ID
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
        model_path = self.model_dir / f"{model_id}.joblib"

        # Package model artifacts
        model_artifacts = {
            "sklearn_model": sklearn_model,
            "statsmodels_model": statsmodels_model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "metrics": {
                "r2": r2,
                "adjusted_r2": adjusted_r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "durbin_watson": dw_stat,
                "breusch_pagan_pvalue": bp_test[1]
            },
            "coefficients": coefficients,
            "p_values": p_values,
            "training_metadata": {
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(feature_columns),
                "created_at": datetime.now().isoformat(),
                "model_id": model_id
            }
        }

        # Serialize model
        async with self._lock:
            joblib.dump(model_artifacts, model_path)
            logger.info(f"Model saved to {model_path}")

        training_time = time.time() - start_time

        # Build response
        response = TrainingResponse(
            model_id=model_id,
            metrics={
                "r2": r2,
                "adjusted_r2": adjusted_r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "durbin_watson": dw_stat,
                "breusch_pagan_pvalue": bp_test[1]
            },
            coefficients=coefficients,
            p_values=p_values,
            r_squared=r2,
            adjusted_r_squared=adjusted_r2,
            mse=mse,
            rmse=rmse,
            mae=mae,
            training_samples=len(X_train),
            test_samples=len(X_test),
            feature_count=len(feature_columns),
            training_time=training_time,
            created_at=datetime.now(),
            model_path=str(model_path)
        )

        logger.info(f"Training completed in {training_time:.2f} seconds")
        return response

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a serialized model."""
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        
        async with self._lock:
            model_artifacts = joblib.load(model_path)
            logger.info(f"Model {model_id} loaded successfully")
            return model_artifacts

    async def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        model_artifacts = await self.load_model(model_id)
        sklearn_model = model_artifacts["sklearn_model"]
        feature_columns = model_artifacts["feature_columns"]
        
        # Validate features
        missing_cols = set(feature_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = data[feature_columns]
        predictions = sklearn_model.predict(X)
        logger.info(f"Made {len(predictions)} predictions with model {model_id}")
        return predictions


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
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file with training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion"),
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
                detail=f"Unsupported file extension. Supported: {SUPPORTED_EXTENSIONS}"
            )

        # Read file content
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )

        # Parse CSV
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

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
            raise HTTPException(status_code=422, detail=e.errors())

        # Train model
        try:
            response = await model_manager.train_model(
                data=data,
                target_column=request.target_column,
                feature_columns=request.feature_columns,
                test_size=request.test_size,
                random_state=request.random_state
            )
            return response
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal training error")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict")
async def predict_endpoint(
    model_id: str = File(..., description="Model ID"),
    file: UploadFile = File(..., description="CSV file with prediction data")
) -> Dict[str, Any]:
    """Make predictions using a trained model."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension. Supported: {SUPPORTED_EXTENSIONS}"
            )

        # Read file content
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )

        # Parse CSV
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

        # Make predictions
        try:
            predictions = await model_manager.predict(model_id=model_id, data=data)
            return {
                "model_id": model_id,
                "predictions": predictions.tolist(),
                "count": len(predictions),
                "timestamp": datetime.now().isoformat()
            }
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal prediction error")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a trained model."""
    try:
        model_artifacts = await model_manager.load_model(model_id)
        return {
            "model_id": model_id,
            "feature_columns": model_artifacts["feature_columns"],
            "target_column": model_artifacts["target_column"],
            "metrics": model_artifacts["metrics"],
            "coefficients": model_artifacts["coefficients"],
            "p_values": model_artifacts["p_values"],
            "training_metadata": model_artifacts["training_metadata"]
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all available models."""
    try:
        model_files = list(self.model_dir.glob("*.joblib")) if hasattr(self, 'model_dir') else list(MODEL_DIR.glob("*.joblib"))
        model_ids = [f.stem for f in model_files]
        return {"models": model_ids, "count": len(model_ids)}
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5
