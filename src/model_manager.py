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


# =============================================================================
# Schemas (Pydantic models)
# =============================================================================

class TrainingRequest(BaseModel):
    """Validated training request parameters."""
    target_column: str = Field(..., min_length=1, description="Name of target variable")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(
        None, description="List of feature columns. If None, all except target are used."
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
        return v


class TrainingResponse(BaseModel):
    """Response model for training results."""
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
    data_shape: Dict[str, int]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: str


# =============================================================================
# Model Manager
# =============================================================================

class RegressionModelManager:
    """Manages training, serialization, and diagnostics for regression models."""

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
    ) -> TrainingResponse:
        """Train a regression model with full diagnostics."""
        start_time = time.time()
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"

        try:
            # Validate data
            if data.empty:
                raise ValueError("Data cannot be empty")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            if len(data) < 10:
                raise ValueError("Data must have at least 10 rows for meaningful training")

            # Prepare features
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            else:
                missing_cols = set(feature_columns) - set(data.columns)
                if missing_cols:
                    raise ValueError(f"Missing feature columns: {missing_cols}")

            # Extract X and y
            X = data[feature_columns].copy()
            y = data[target_column].copy()

            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning("Missing values detected, dropping rows")
                mask = X.notnull().all(axis=1) & y.notnull()
                X = X[mask]
                y = y[mask]

            # Validate numeric data
            if not np.issubdtype(X.dtypes.iloc[0], np.number):
                raise ValueError("Features must be numeric")
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("Target must be numeric")

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
            y_pred = sklearn_model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            # Adjusted R²
            n = len(y_test)
            p = X_test.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

            # Coefficients and p-values
            coefficients = dict(zip(feature_columns, sklearn_model.coef_))
            p_values = {}
            for idx, col in enumerate(feature_columns):
                p_values[col] = float(stats_model.pvalues[idx + 1])  # +1 for constant

            # Feature importance (absolute coefficients normalized)
            abs_coefs = np.abs(sklearn_model.coef_)
            feature_importance = {
                col: float(abs_coefs[i] / abs_coefs.sum())
                for i, col in enumerate(feature_columns)
            }

            # Diagnostics
            diagnostics = self._compute_diagnostics(stats_model, X_train, y_train)

            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            model_data = {
                "sklearn_model": sklearn_model,
                "statsmodels_model": stats_model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "training_metadata": {
                    "model_id": model_id,
                    "timestamp": datetime.now().isoformat(),
                    "test_size": test_size,
                    "random_state": random_state,
                    "n_samples": len(data),
                    "n_features": len(feature_columns)
                }
            }
            await self._save_model(model_data, model_path)

            training_time = time.time() - start_time
            logger.info(f"Model {model_id} trained in {training_time:.2f}s")

            return TrainingResponse(
                model_id=model_id,
                training_timestamp=datetime.now().isoformat(),
                metrics={
                    "r2": float(r2),
                    "adjusted_r2": float(adjusted_r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "training_time": training_time
                },
                coefficients=coefficients,
                p_values=p_values,
                r_squared=float(r2),
                adjusted_r_squared=float(adjusted_r2),
                mse=float(mse),
                rmse=float(rmse),
                mae=float(mae),
                feature_importance=feature_importance,
                diagnostics=diagnostics,
                model_path=str(model_path),
                data_shape={"rows": len(data), "features": len(feature_columns)}
            )

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    async def _save_model(self, model_data: Dict[str, Any], path: Path) -> None:
        """Asynchronously save model using joblib."""
        async with self._lock:
            try:
                # Run in thread pool to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    None, joblib.dump, model_data, path
                )
                logger.info(f"Model saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")
                raise

    def _compute_diagnostics(
        self,
        stats_model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Compute statistical diagnostics for the model."""
        try:
            # Residuals
            residuals = stats_model.resid
            fitted = stats_model.fittedvalues

            # Normality test (Shapiro-Wilk)
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            # Heteroscedasticity (Breusch-Pagan)
            bp_test = het_breuschpagan(residuals, stats_model.model.exog)
            bp_stat, bp_p, bp_f, bp_f_p = bp_test

            # Autocorrelation (Durbin-Watson)
            dw_stat = durbin_watson(residuals)

            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(residuals)

            return {
                "residuals_mean": float(np.mean(residuals)),
                "residuals_std": float(np.std(residuals)),
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p)
                },
                "breusch_pagan": {
                    "statistic": float(bp_stat),
                    "p_value": float(bp_p),
                    "f_statistic": float(bp_f),
                    "f_p_value": float(bp_f_p)
                },
                "durbin_watson": float(dw_stat),
                "jarque_bera": {
                    "statistic": float(jb_stat),
                    "p_value": float(jb_p)
                },
                "aic": float(stats_model.aic),
                "bic": float(stats_model.bic),
                "f_statistic": float(stats_model.fvalue),
                "f_p_value": float(stats_model.f_pvalue)
            }
        except Exception as e:
            logger.warning(f"Could not compute all diagnostics: {str(e)}")
            return {"error": str(e)}


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Regression API Service",
    description="Production-grade regression model training and serving API",
    version="1.0.0"
)

model_manager = RegressionModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "regression-api-service"
    }


@app.post("/train", response_model=TrainingResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def train_model(
    file: UploadFile = File(..., description="CSV file with training data"),
    target_column: str = File(..., description="Target column name"),
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
            logger.warning(f"Validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        # Read and validate CSV
        try:
            content = await file.read()
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to read CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

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
        logger.error(f"Unexpected error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for consistent error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            detail=str(exc.detail),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# Import for JSONResponse
from fastapi.responses import JSONResponse


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3
