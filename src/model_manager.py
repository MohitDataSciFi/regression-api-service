import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError
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

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)


# ==================== Schemas ====================
class TrainingRequest(BaseModel):
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns (default: all except target)")

    class Config:
        json_schema_extra = {
            "example": {
                "target_column": "price",
                "test_size": 0.2,
                "random_state": 42,
                "feature_columns": ["sqft", "bedrooms", "bathrooms"]
            }
        }


class TrainingResponse(BaseModel):
    model_id: str
    training_timestamp: datetime
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    mse: float
    mae: float
    rmse: float
    durbin_watson: float
    breusch_pagan_pvalue: float
    feature_importance: Dict[str, float]
    model_path: str
    sample_size: int
    n_features: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime


# ==================== Model Manager ====================
@dataclass
class ModelArtifacts:
    model_id: str
    sklearn_model: LinearRegression
    statsmodels_model: Any
    feature_names: List[str]
    target_name: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    model_path: str


class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelArtifacts] = {}
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_from_csv(
        self,
        file_content: bytes,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None
    ) -> TrainingResponse:
        """Train a regression model from CSV data asynchronously."""
        try:
            # Read CSV data
            df = await asyncio.to_thread(self._read_csv, file_content)
            logger.info(f"Loaded dataset with shape: {df.shape}")

            # Validate target column
            if target_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")

            # Determine feature columns
            if feature_columns:
                missing_cols = set(feature_columns) - set(df.columns)
                if missing_cols:
                    raise HTTPException(status_code=400, detail=f"Missing feature columns: {missing_cols}")
                X = df[feature_columns]
            else:
                X = df.drop(columns=[target_column])

            y = df[target_column]

            # Validate data types
            if not np.issubdtype(y.dtype, np.number):
                raise HTTPException(status_code=400, detail="Target column must be numeric")

            # Check for non-numeric features
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                raise HTTPException(status_code=400, detail=f"Non-numeric feature columns: {non_numeric}")

            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning("Missing values detected, dropping rows")
                valid_mask = X.notnull().all(axis=1) & y.notnull()
                X = X[valid_mask]
                y = y[valid_mask]

            # Train model in thread pool
            model_artifacts = await asyncio.to_thread(
                self._train_model,
                X, y, target_column, test_size, random_state
            )

            # Store model
            self._models[model_artifacts.model_id] = model_artifacts

            # Build response
            response = TrainingResponse(
                model_id=model_artifacts.model_id,
                training_timestamp=datetime.utcnow(),
                metrics=model_artifacts.metrics,
                coefficients=model_artifacts.coefficients,
                p_values=model_artifacts.p_values,
                r_squared=model_artifacts.metrics["r_squared"],
                adjusted_r_squared=model_artifacts.metrics["adjusted_r_squared"],
                mse=model_artifacts.metrics["mse"],
                mae=model_artifacts.metrics["mae"],
                rmse=model_artifacts.metrics["rmse"],
                durbin_watson=model_artifacts.metrics["durbin_watson"],
                breusch_pagan_pvalue=model_artifacts.metrics["breusch_pagan_pvalue"],
                feature_importance=model_artifacts.metrics["feature_importance"],
                model_path=model_artifacts.model_path,
                sample_size=len(X),
                n_features=X.shape[1]
            )

            logger.info(f"Model trained successfully: {model_artifacts.model_id}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    def _read_csv(self, file_content: bytes) -> pd.DataFrame:
        """Read CSV from bytes."""
        try:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            df = pd.read_csv(tmp_path)
            Path(tmp_path).unlink()
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str,
        test_size: float,
        random_state: int
    ) -> ModelArtifacts:
        """Train regression model with diagnostics."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train sklearn model
        sklearn_model = LinearRegression()
        sklearn_model.fit(X_train, y_train)

        # Train statsmodels for diagnostics
        X_train_const = add_constant(X_train)
        statsmodels_model = OLS(y_train, X_train_const).fit()

        # Predictions
        y_pred = sklearn_model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Adjusted R²
        n = len(X_test)
        p = X_test.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Diagnostics
        residuals = y_test - y_pred
        dw_stat = durbin_watson(residuals)

        # Breusch-Pagan test
        try:
            bp_test = het_breuschpagan(residuals, add_constant(X_test))
            bp_pvalue = bp_test[1]
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            bp_pvalue = float("nan")

        # Coefficients and p-values
        coefficients = {}
        p_values = {}
        feature_importance = {}

        for i, col in enumerate(X.columns):
            coefficients[col] = float(sklearn_model.coef_[i])
            p_values[col] = float(statsmodels_model.pvalues[col])
            # Use absolute coefficient as importance proxy
            feature_importance[col] = float(abs(sklearn_model.coef_[i]))

        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v / total_importance for k, v in feature_importance.items()}

        # Generate model ID
        model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

        # Save model
        model_path = self.model_dir / f"{model_id}.joblib"
        joblib.dump({
            "model": sklearn_model,
            "feature_names": X.columns.tolist(),
            "target_name": target_name,
            "model_id": model_id,
            "training_date": datetime.utcnow().isoformat()
        }, model_path)

        metrics = {
            "r_squared": float(r2),
            "adjusted_r_squared": float(adjusted_r2),
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "durbin_watson": float(dw_stat),
            "breusch_pagan_pvalue": float(bp_pvalue),
            "feature_importance": feature_importance
        }

        logger.info(f"Model trained with R²={r2:.4f}, RMSE={rmse:.4f}")

        return ModelArtifacts(
            model_id=model_id,
            sklearn_model=sklearn_model,
            statsmodels_model=statsmodels_model,
            feature_names=X.columns.tolist(),
            target_name=target_name,
            metrics=metrics,
            coefficients=coefficients,
            p_values=p_values,
            model_path=str(model_path)
        )

    def get_model(self, model_id: str) -> Optional[ModelArtifacts]:
        """Retrieve a trained model by ID."""
        return self._models.get(model_id)

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a serialized model."""
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")


# ==================== API ====================
app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0"
)

model_manager = ModelManager()


@app.post("/train", response_model=TrainingResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def train_model(
    file: UploadFile = File(..., description="CSV file for training"),
    target_column: str = File(..., description="Target column name"),
    test_size: float = File(0.2, ge=0.1, le=0.5, description="Test set proportion"),
    random_state: int = File(42, ge=0, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
):
    """
    Train a regression model from uploaded CSV data.
    
    - **file**: CSV file containing training data
    - **target_column**: Name of the target (dependent) variable
    - **test_size**: Proportion of data to use for testing (0.1-0.5)
    - **random_state**: Random seed for reproducibility
    - **feature_columns**: Optional comma-separated list of feature columns
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Parse feature columns if provided
        feature_cols = None
        if feature_columns:
            feature_cols = [col.strip() for col in feature_columns.split(",") if col.strip()]

        # Train model
        response = await model_manager.train_from_csv(
            file_content=file_content,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            feature_columns=feature_cols
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": len(model_manager._models)
    }


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a trained model."""
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "model_id": model.model_id,
        "feature_names": model.feature_names,
        "target_name": model.target_name,
        "metrics": model.metrics,
        "model_path": model.model_path
    }


@app.get("/models")
async def list_models():
    """List all trained models."""
    return {
        "models": [
            {
                "model_id": model_id,
                "target": model.target_name,
                "features": model.feature_names,
                "r_squared": model.metrics["r_squared"],
                "created": model.model_id.split("_")[1] if len(model.model_id.split("_")) > 1 else "unknown"
            }
            for model_id, model in model_manager._models.items()
        ]
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            detail=str(exc.detail),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.utcnow()
        ).dict()
    )


from fastapi.responses import JSONResponse


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Regression API Service")
    # Load any existing models from disk
    for model_file in model_manager.model_dir.glob("*.joblib"):
        try:
            model_data = joblib.load(model_file)
            logger.info(f"Loaded existing model: {model_data.get('model_id', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to load model {model_file}: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Regression API Service")
    executor.shutdown(wait=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")