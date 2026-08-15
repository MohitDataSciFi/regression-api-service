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
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
SUPPORTED_EXTENSIONS = {".csv", ".txt"}


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
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
    n_samples: int
    n_features: int
    feature_columns: List[str]
    target_column: str
    model_path: str
    diagnostics: Dict[str, Any]


class ModelManager:
    """Manages model training, serialization, and diagnostics."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None
    ) -> TrainingResponse:
        """Train an OLS regression model with diagnostics."""
        async with self._lock:
            try:
                logger.info(f"Starting model training with target: {target_column}")
                start_time = time.time()

                # Validate data
                self._validate_data(data, target_column, feature_columns)

                # Prepare features and target
                if feature_columns is None:
                    feature_columns = [col for col in data.columns if col != target_column]
                
                X = data[feature_columns].copy()
                y = data[target_column].copy()

                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    logger.warning("Missing values detected, dropping rows with NaN")
                    mask = X.notnull().all(axis=1) & y.notnull()
                    X = X[mask]
                    y = y[mask]

                # Convert to numeric if possible
                X = X.apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(y, errors='coerce')

                # Drop any remaining NaN after conversion
                mask = X.notnull().all(axis=1) & y.notnull()
                X = X[mask]
                y = y[mask]

                if len(X) < 10:
                    raise ValueError("Insufficient data after cleaning. Need at least 10 samples.")

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

                # Extract coefficients and p-values
                coefficients = dict(zip(feature_columns, sklearn_model.coef_))
                coefficients["intercept"] = float(sklearn_model.intercept_)

                p_values = {}
                for i, col in enumerate(feature_columns):
                    p_values[col] = float(stats_model.pvalues[i + 1])  # +1 for constant
                p_values["intercept"] = float(stats_model.pvalues[0])

                # Calculate diagnostics
                diagnostics = self._calculate_diagnostics(stats_model, X_train, y_train)

                # Generate model ID
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
                model_path = self.model_dir / f"{model_id}.joblib"

                # Serialize model
                model_artifacts = {
                    "sklearn_model": sklearn_model,
                    "statsmodels_model": stats_model,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "model_id": model_id,
                    "training_timestamp": datetime.now().isoformat(),
                    "metrics": metrics,
                    "coefficients": coefficients,
                    "p_values": p_values,
                    "diagnostics": diagnostics
                }

                joblib.dump(model_artifacts, model_path)
                logger.info(f"Model saved to {model_path}")

                # Build response
                response = TrainingResponse(
                    model_id=model_id,
                    training_timestamp=datetime.now().isoformat(),
                    metrics=metrics,
                    coefficients=coefficients,
                    p_values=p_values,
                    r_squared=float(stats_model.rsquared),
                    adjusted_r_squared=float(stats_model.rsquared_adj),
                    mse=float(metrics["test_mse"]),
                    rmse=float(metrics["test_rmse"]),
                    mae=float(metrics["test_mae"]),
                    n_samples=len(X),
                    n_features=len(feature_columns),
                    feature_columns=feature_columns,
                    target_column=target_column,
                    model_path=str(model_path),
                    diagnostics=diagnostics
                )

                elapsed_time = time.time() - start_time
                logger.info(f"Model training completed in {elapsed_time:.2f} seconds")
                return response

            except Exception as e:
                logger.error(f"Model training failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    def _validate_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Data is empty")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        if feature_columns is not None:
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")

        # Check for non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column not in numeric_cols:
            raise ValueError(f"Target column '{target_column}' must be numeric")

        if feature_columns is not None:
            non_numeric = [col for col in feature_columns if col not in numeric_cols]
            if non_numeric:
                raise ValueError(f"Non-numeric feature columns: {non_numeric}")

    def _calculate_diagnostics(
        self,
        stats_model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Calculate statistical diagnostics for the model."""
        try:
            # Residual diagnostics
            residuals = stats_model.resid
            fitted = stats_model.fittedvalues

            # Normality test (Shapiro-Wilk)
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            # Heteroscedasticity test (Breusch-Pagan)
            bp_test = het_breuschpagan(residuals, stats_model.model.exog)
            bp_stat, bp_p, bp_fstat, bp_fp = bp_test

            # Autocorrelation (Durbin-Watson)
            dw_stat = durbin_watson(residuals)

            # Residual statistics
            residual_stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals))
            }

            # F-statistic and p-value
            f_stat = float(stats_model.fvalue)
            f_p_value = float(stats_model.f_pvalue)

            # AIC and BIC
            aic = float(stats_model.aic)
            bic = float(stats_model.bic)

            return {
                "shapiro_wilk_statistic": float(shapiro_stat),
                "shapiro_wilk_p_value": float(shapiro_p),
                "breusch_pagan_statistic": float(bp_stat),
                "breusch_pagan_p_value": float(bp_p),
                "durbin_watson": float(dw_stat),
                "residual_stats": residual_stats,
                "f_statistic": f_stat,
                "f_p_value": f_p_value,
                "aic": aic,
                "bic": bic,
                "degrees_of_freedom": int(stats_model.df_model),
                "residual_degrees_of_freedom": int(stats_model.df_resid)
            }
        except Exception as e:
            logger.warning(f"Could not calculate all diagnostics: {str(e)}")
            return {"error": str(e)}


# Initialize FastAPI app
app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0"
)

# Initialize model manager
model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=TrainingResponse)
async def train_model_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
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
                detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
            )

        # Read file content
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)} MB"
            )

        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
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
            raise HTTPException(status_code=422, detail=str(e))

        # Train model
        response = await model_manager.train_model(
            data=df,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            feature_columns=request.feature_columns
        )

        return response

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

        model_artifacts = joblib.load(model_path)
        return {
            "model_id": model_artifacts["model_id"],
            "training_timestamp": model_artifacts["training_timestamp"],
            "metrics": model_artifacts["metrics"],
            "coefficients": model_artifacts["coefficients"],
            "p_values": model_artifacts["p_values"],
            "feature_columns": model_artifacts["feature_columns"],
            "target_column": model_artifacts["target_column"],
            "diagnostics": model_artifacts["diagnostics"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/models")
async def list_models() -> List[Dict[str, Any]]:
    """List all trained models."""
    try:
        models = []
        for model_file in MODEL_DIR.glob("*.joblib"):
            try:
                model_artifacts = joblib.load(model_file)
                models.append({
                    "model_id": model_artifacts["model_id"],
                    "training_timestamp": model_artifacts["training_timestamp"],
                    "target_column": model_artifacts["target_column"],
                    "n_features": len(model_artifacts["feature_columns"]),
                    "test_r2": model_artifacts["metrics"]["test_r2"]
                })
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {str(e)}")
        
        return sorted(models, key=lambda x: x["training_timestamp"], reverse=True)
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.delete("/models/{model_id}")
async def delete_model(model_id: str) -> Dict[str, str]:
    """Delete a trained model."""
    try:
        model_path = MODEL_DIR / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        model_path.unlink()
        logger.info(f"Deleted model {model_id}")
        return {"status": "deleted", "model_id": model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")