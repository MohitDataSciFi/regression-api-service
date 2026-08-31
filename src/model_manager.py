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
from pydantic import BaseModel, Field, field_validator
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
    model_name: str = Field("regression_model", min_length=1, description="Model identifier")

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        if not v.replace("_", "").isalnum():
            raise ValueError("Model name must be alphanumeric with underscores only")
        return v.strip()


class TrainingResponse(BaseModel):
    """Response schema for training results."""
    model_id: str
    model_path: str
    metrics: Dict[str, Any]
    diagnostics: Dict[str, Any]
    training_timestamp: str
    training_duration: float


class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with directory: {self.model_dir}")

    async def train_model(
        self,
        file_content: bytes,
        target_column: str,
        test_size: float,
        random_state: int,
        model_name: str
    ) -> TrainingResponse:
        """Train a regression model with full diagnostics."""
        start_time = time.time()
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Parse and validate data
            df = await self._parse_csv(file_content)
            await self._validate_data(df, target_column)

            # Prepare features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)

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

            # Compute metrics
            metrics = await self._compute_metrics(sklearn_model, X_test, y_test)
            diagnostics = await self._compute_diagnostics(statsmodels_model, X_train_const, y_train)

            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            await self._serialize_model(
                model_path=model_path,
                model=sklearn_model,
                feature_names=list(X.columns),
                target_column=target_column,
                metrics=metrics,
                diagnostics=diagnostics
            )

            training_duration = time.time() - start_time
            logger.info(f"Model {model_id} trained successfully in {training_duration:.2f}s")

            return TrainingResponse(
                model_id=model_id,
                model_path=str(model_path),
                metrics=metrics,
                diagnostics=diagnostics,
                training_timestamp=datetime.now().isoformat(),
                training_duration=training_duration
            )

        except Exception as e:
            logger.error(f"Training failed for model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    async def _parse_csv(self, file_content: bytes) -> pd.DataFrame:
        """Parse CSV content with validation."""
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            if df.empty:
                raise ValueError("CSV file is empty")
            if df.shape[1] < 2:
                raise ValueError("CSV must contain at least 2 columns (features and target)")
            return df
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error: {str(e)}")

    async def _validate_data(self, df: pd.DataFrame, target_column: str) -> None:
        """Validate data quality and target column."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            missing_cols = missing[missing > 0].index.tolist()
            raise ValueError(f"Missing values found in columns: {missing_cols}")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if np.isinf(df[numeric_cols]).any().any():
            raise ValueError("Infinite values found in numeric columns")

        # Check target is numeric
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise ValueError(f"Target column '{target_column}' must be numeric")

        # Check for sufficient data
        if len(df) < 10:
            raise ValueError("Insufficient data for training (minimum 10 samples)")

    async def _compute_metrics(
        self,
        model: LinearRegression,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Compute model performance metrics."""
        predictions = model.predict(X_test)

        metrics = {
            "r2_score": float(r2_score(y_test, predictions)),
            "mean_absolute_error": float(mean_absolute_error(y_test, predictions)),
            "mean_squared_error": float(mean_squared_error(y_test, predictions)),
            "root_mean_squared_error": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "num_features": int(X_test.shape[1]),
            "num_samples": int(len(y_test))
        }

        # Add R² adjusted
        n = len(y_test)
        p = X_test.shape[1]
        metrics["adjusted_r2"] = float(
            1 - (1 - metrics["r2_score"]) * (n - 1) / (n - p - 1)
        )

        return metrics

    async def _compute_diagnostics(
        self,
        model: OLS,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Compute statistical diagnostics from statsmodels."""
        try:
            # Coefficient statistics
            coefficients = {}
            for idx, coef in enumerate(model.params):
                col_name = X_train.columns[idx]
                coefficients[col_name] = {
                    "coefficient": float(coef),
                    "std_error": float(model.bse[idx]),
                    "t_statistic": float(model.tvalues[idx]),
                    "p_value": float(model.pvalues[idx]),
                    "confidence_interval": [
                        float(model.conf_int()[idx][0]),
                        float(model.conf_int()[idx][1])
                    ]
                }

            # Model diagnostics
            residuals = model.resid
            n = len(residuals)
            k = X_train.shape[1]

            # Normality test (Shapiro-Wilk)
            shapiro_stat, shapiro_p = stats.shapiro(residuals)

            # Heteroscedasticity test (Breusch-Pagan)
            try:
                bp_test = het_breuschpagan(residuals, X_train)
                bp_stat, bp_p, bp_f, bp_f_p = bp_test
            except Exception:
                bp_stat, bp_p, bp_f, bp_f_p = None, None, None, None

            # Autocorrelation test (Durbin-Watson)
            dw_stat = durbin_watson(residuals)

            diagnostics = {
                "coefficients": coefficients,
                "model_summary": {
                    "r_squared": float(model.rsquared),
                    "adjusted_r_squared": float(model.rsquared_adj),
                    "f_statistic": float(model.fvalue),
                    "f_p_value": float(model.f_pvalue),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "log_likelihood": float(model.llf),
                    "degrees_of_freedom": int(model.df_model),
                    "residual_degrees_of_freedom": int(model.df_resid)
                },
                "residual_diagnostics": {
                    "shapiro_wilk_statistic": float(shapiro_stat),
                    "shapiro_wilk_p_value": float(shapiro_p),
                    "durbin_watson_statistic": float(dw_stat),
                    "breusch_pagan_statistic": float(bp_stat) if bp_stat else None,
                    "breusch_pagan_p_value": float(bp_p) if bp_p else None,
                    "breusch_pagan_f_statistic": float(bp_f) if bp_f else None,
                    "breusch_pagan_f_p_value": float(bp_f_p) if bp_f_p else None
                }
            }

            return diagnostics

        except Exception as e:
            logger.warning(f"Diagnostics computation partially failed: {str(e)}")
            return {
                "coefficients": {},
                "model_summary": {},
                "residual_diagnostics": {
                    "error": str(e)
                }
            }

    async def _serialize_model(
        self,
        model_path: Path,
        model: LinearRegression,
        feature_names: List[str],
        target_column: str,
        metrics: Dict[str, Any],
        diagnostics: Dict[str, Any]
    ) -> None:
        """Serialize model and metadata using joblib."""
        model_package = {
            "model": model,
            "feature_names": feature_names,
            "target_column": target_column,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "created_at": datetime.now().isoformat(),
            "model_type": "linear_regression"
        }

        # Use lock to prevent concurrent writes
        async with self._lock:
            # Write to temp file first, then rename for atomicity
            temp_path = model_path.with_suffix(".tmp")
            joblib.dump(model_package, temp_path)
            temp_path.rename(model_path)

        logger.info(f"Model serialized to {model_path}")


# Initialize FastAPI app and model manager
app = FastAPI(
    title="Regression API Service",
    description="Production-grade regression model training and serving API",
    version="1.0.0"
)
model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=TrainingResponse)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file for training"),
    target_column: str = File(..., description="Target column name"),
    test_size: float = File(0.2, description="Test set proportion"),
    random_state: int = File(42, description="Random seed"),
    model_name: str = File("regression_model", description="Model identifier")
) -> TrainingResponse:
    """Train a regression model from uploaded CSV data."""
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Read and validate file size
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )

    # Validate request parameters
    try:
        request = TrainingRequest(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            model_name=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Train model
    try:
        result = await model_manager.train_model(
            file_content=file_content,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            model_name=request.model_name
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """Retrieve model information and metadata."""
    model_path = MODEL_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    try:
        model_package = joblib.load(model_path)
        return {
            "model_id": model_id,
            "model_type": model_package.get("model_type"),
            "feature_names": model_package.get("feature_names"),
            "target_column": model_package.get("target_column"),
            "metrics": model_package.get("metrics"),
            "diagnostics": model_package.get("diagnostics"),
            "created_at": model_package.get("created_at")
        }
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all available trained models."""
    model_files = list(MODEL_DIR.glob("*.joblib"))
    model_ids = [f.stem for f in model_files]
    return {"models": model_ids}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5

# Phase 1: Core Model Training and Serialization - iteration 6

# Phase 1: Core Model Training and Serialization - iteration 7

# Phase 1: Core Model Training and Serialization - iteration 8
