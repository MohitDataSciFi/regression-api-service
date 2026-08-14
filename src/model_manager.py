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
        logging.FileHandler("regression_service.log"),
        logging.StreamHandler()
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
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns. If None, all other columns are used.")

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
    feature_count: int
    sample_count: int
    model_path: str
    diagnostics: Dict[str, Any]


class ModelManager:
    """Manages regression model training, serialization, and diagnostics."""

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
    ) -> TrainingResponse:
        """Train an OLS regression model with full diagnostics."""
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

                # Handle categorical variables
                X = pd.get_dummies(X, drop_first=True)
                logger.info(f"Feature matrix shape after encoding: {X.shape}")

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

                # Train sklearn model
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)

                # Train statsmodels for diagnostics
                X_train_const = add_constant(X_train)
                stats_model = OLS(y_train, X_train_const).fit()

                # Make predictions
                y_pred_train = sklearn_model.predict(X_train)
                y_pred_test = sklearn_model.predict(X_test)

                # Compute metrics
                metrics = self._compute_metrics(y_test, y_pred_test, y_train, y_pred_train)

                # Extract coefficients and p-values
                coefficients = self._extract_coefficients(sklearn_model, X.columns)
                p_values = self._extract_p_values(stats_model, X.columns)

                # Compute diagnostics
                diagnostics = self._compute_diagnostics(stats_model, X_train_const, y_train)

                # Generate model ID and save
                model_id = f"reg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
                model_path = self._save_model(sklearn_model, model_id, X.columns.tolist())

                # Prepare response
                response = TrainingResponse(
                    model_id=model_id,
                    training_timestamp=datetime.now().isoformat(),
                    metrics=metrics,
                    coefficients=coefficients,
                    p_values=p_values,
                    r_squared=metrics["r2"],
                    adjusted_r_squared=metrics["adjusted_r2"],
                    mse=metrics["mse"],
                    rmse=metrics["rmse"],
                    mae=metrics["mae"],
                    feature_count=len(X.columns),
                    sample_count=len(X),
                    model_path=str(model_path),
                    diagnostics=diagnostics
                )

                elapsed_time = time.time() - start_time
                logger.info(f"Model training completed in {elapsed_time:.2f} seconds. Model ID: {model_id}")
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
        """Validate input data for training."""
        if data.empty:
            raise ValueError("Input data is empty")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        if feature_columns is not None:
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")

        # Check for non-numeric target
        if not pd.api.types.is_numeric_dtype(data[target_column]):
            raise ValueError("Target column must be numeric")

        # Check for sufficient data
        if len(data) < 10:
            raise ValueError("Insufficient data for training (minimum 10 samples)")

        # Check for NaN values
        if data.isnull().any().any():
            logger.warning("Data contains NaN values. Dropping rows with NaN.")
            data.dropna(inplace=True)

    def _compute_metrics(
        self,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        y_train: np.ndarray,
        y_pred_train: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive regression metrics."""
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        # Adjusted R²
        n = len(y_test)
        p = 1  # number of predictors (simplified)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Training metrics
        train_r2 = r2_score(y_train, y_pred_train)
        train_mse = mean_squared_error(y_train, y_pred_train)

        return {
            "r2": float(r2),
            "adjusted_r2": float(adjusted_r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "train_r2": float(train_r2),
            "train_mse": float(train_mse)
        }

    def _extract_coefficients(
        self,
        model: LinearRegression,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract model coefficients."""
        coefficients = {}
        for name, coef in zip(feature_names, model.coef_):
            coefficients[name] = float(coef)
        coefficients["intercept"] = float(model.intercept_)
        return coefficients

    def _extract_p_values(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract p-values from statsmodels model."""
        p_values = {}
        for name, pval in zip(model.pvalues.index, model.pvalues):
            if name == "const":
                p_values["intercept"] = float(pval)
            else:
                p_values[name] = float(pval)
        return p_values

    def _compute_diagnostics(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Compute model diagnostics."""
        diagnostics = {}
        
        try:
            # Durbin-Watson test for autocorrelation
            residuals = model.resid
            dw_stat = durbin_watson(residuals)
            diagnostics["durbin_watson"] = float(dw_stat)
            
            # Breusch-Pagan test for heteroscedasticity
            bp_test = het_breuschpagan(residuals, X)
            diagnostics["breusch_pagan_lm"] = float(bp_test[0])
            diagnostics["breusch_pagan_pvalue"] = float(bp_test[1])
            
            # Jarque-Bera test for normality of residuals
            jb_stat, jb_pvalue, skew, kurtosis = stats.jarque_bera(residuals)
            diagnostics["jarque_bera_stat"] = float(jb_stat)
            diagnostics["jarque_bera_pvalue"] = float(jb_pvalue)
            diagnostics["skewness"] = float(skew)
            diagnostics["kurtosis"] = float(kurtosis)
            
            # F-statistic
            diagnostics["f_statistic"] = float(model.fvalue)
            diagnostics["f_pvalue"] = float(model.f_pvalue)
            
            # AIC and BIC
            diagnostics["aic"] = float(model.aic)
            diagnostics["bic"] = float(model.bic)
            
        except Exception as e:
            logger.warning(f"Could not compute some diagnostics: {str(e)}")
            diagnostics["error"] = str(e)
        
        return diagnostics

    def _save_model(
        self,
        model: LinearRegression,
        model_id: str,
        feature_names: List[str]
    ) -> Path:
        """Save model and metadata to disk."""
        model_path = self.model_dir / f"{model_id}.joblib"
        
        # Save model with metadata
        model_data = {
            "model": model,
            "feature_names": feature_names,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "model_type": "linear_regression"
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a saved model."""
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        try:
            model_data = joblib.load(model_path)
            logger.info(f"Model {model_id} loaded successfully")
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0"
)

# Initialize model manager
model_manager = ModelManager()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "regression-api-service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_count": len(list(MODEL_DIR.glob("*.joblib"))),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
):
    """
    Train a regression model from uploaded CSV data.
    
    Args:
        file: CSV file with training data
        target_column: Name of the target column
        test_size: Proportion of data for testing (0.1-0.5)
        random_state: Random seed for reproducibility
        feature_columns: Comma-separated list of feature columns (optional)
    
    Returns:
        TrainingResponse with model metrics and diagnostics
    """
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
            )

        # Read and validate file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )

        # Parse CSV data
        try:
            data = pd.read_csv(io.BytesIO(content))
            logger.info(f"Loaded data with shape: {data.shape}")
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
            data=data,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            random_state=request.random_state
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a trained model."""
    try:
        model_data = await model_manager.load_model(model_id)
        return {
            "model_id": model_data["model_id"],
            "model_type": model_data["model_type"],
            "feature_names": model_data["feature_names"],
            "created_at": model_data["created_at"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")


@app.get("/models")
async def list_models():
    """List all trained models."""
    try:
        model_files = list(MODEL_DIR.glob("*.joblib"))
        models = []
        for model_file in model_files:
            try:
                model_data = joblib.load(model_file)
                models.append({
                    "model_id": model_data["model_id"],
                    "model_type": model_data["model_type"],
                    "created_at": model_data["created_at"],
                    "feature_count": len(model_data["feature_names"])
                })
            except Exception as e:
                logger.warning(f"Failed to load model metadata from {model_file}: {str(e)}")
        
        return {"models": models, "total": len(models)}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.post("/predict/{model_id}")
async def predict(
    model_id: str,
    file: UploadFile = File(..., description="CSV file with features for prediction")
):
    """Make predictions using a trained model."""
    try:
        # Load model
        model_data = await model_manager.load_model(model_id)
        model = model_data["model"]
        feature_names = model_data["feature_names"]

        # Read prediction data
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

        # Validate features
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        # Prepare features
        X = data[feature_names].copy()
        X = pd.get_dummies(X, drop_first=True)

        # Make predictions
        predictions = model.predict(X)

        # Prepare response
        response = {
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "sample_count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4
