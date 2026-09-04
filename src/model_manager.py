import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
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

# Constants
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".csv"}


# ==================== Schemas ====================
class TrainingResponse(BaseModel):
    """Response schema for training endpoint."""
    model_id: str
    timestamp: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    diagnostics: Dict[str, Any]
    model_path: str
    feature_columns: List[str]
    target_column: str


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    
    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: float
    model_id: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    timestamp: str


# ==================== Model Manager ====================
class ModelManager:
    """Manages regression model training, serialization, and diagnostics."""
    
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelManager initialized with directory: {self.model_dir}")
    
    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Dict[str, Any], str]:
        """
        Train an OLS regression model with diagnostics.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            feature_columns: List of feature columns (None = all except target)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (metrics_dict, model_id)
        """
        start_time = time.time()
        logger.info(f"Starting model training with target: {target_column}")
        
        try:
            # Validate data
            if data.empty:
                raise ValueError("Input data is empty")
            
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            if not feature_columns:
                raise ValueError("No feature columns available for training")
            
            # Validate feature columns exist
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")
            
            # Prepare data
            X = data[feature_columns].copy()
            y = data[target_column].copy()
            
            # Handle missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning("Missing values detected, dropping rows")
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
            
            # Convert to numeric
            X = X.apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')
            
            # Drop any remaining NaN rows
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                raise ValueError("Insufficient data after cleaning (need at least 10 samples)")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train scikit-learn model
            sklearn_model = LinearRegression()
            sklearn_model.fit(X_train, y_train)
            
            # Train statsmodels for diagnostics
            X_train_const = add_constant(X_train)
            stats_model = OLS(y_train, X_train_const).fit()
            
            # Generate predictions
            y_pred = sklearn_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate adjusted R²
            n = len(X_test)
            p = len(feature_columns)
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            # Extract coefficients and p-values
            coefficients = dict(zip(feature_columns, sklearn_model.coef_))
            coefficients["intercept"] = float(sklearn_model.intercept_)
            
            p_values = {}
            for col in feature_columns:
                if col in stats_model.pvalues:
                    p_values[col] = float(stats_model.pvalues[col])
            p_values["intercept"] = float(stats_model.pvalues.get("const", 0.0))
            
            # Diagnostics
            diagnostics = {
                "f_statistic": float(stats_model.fvalue),
                "f_p_value": float(stats_model.f_pvalue),
                "aic": float(stats_model.aic),
                "bic": float(stats_model.bic),
                "durbin_watson": float(durbin_watson(stats_model.resid)),
                "condition_number": float(np.linalg.cond(X_train_const)),
                "residual_std_error": float(np.sqrt(stats_model.mse_resid)),
                "skewness": float(stats.skew(stats_model.resid)),
                "kurtosis": float(stats.kurtosis(stats_model.resid)),
                "jarque_bera_stat": float(stats.jarque_bera(stats_model.resid)[0]),
                "jarque_bera_p": float(stats.jarque_bera(stats_model.resid)[1]),
            }
            
            # Breusch-Pagan test for heteroscedasticity
            try:
                bp_test = het_breuschpagan(stats_model.resid, X_train_const)
                diagnostics["breusch_pagan_lm"] = float(bp_test[0])
                diagnostics["breusch_pagan_p"] = float(bp_test[1])
            except Exception as e:
                logger.warning(f"Breusch-Pagan test failed: {e}")
                diagnostics["breusch_pagan_lm"] = None
                diagnostics["breusch_pagan_p"] = None
            
            # Prepare metrics
            metrics = {
                "r2": float(r2),
                "adjusted_r2": float(adjusted_r2),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "training_time_seconds": float(time.time() - start_time),
                "n_features": int(len(feature_columns))
            }
            
            # Generate model ID
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
            
            # Package model artifacts
            model_artifacts = {
                "model_id": model_id,
                "sklearn_model": sklearn_model,
                "statsmodels_model": stats_model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "coefficients": coefficients,
                "p_values": p_values,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "training_date": datetime.now().isoformat(),
                "model_version": "1.0.0"
            }
            
            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            joblib.dump(model_artifacts, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Prepare response data
            response_data = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "coefficients": coefficients,
                "p_values": p_values,
                "diagnostics": diagnostics,
                "model_path": str(model_path),
                "feature_columns": feature_columns,
                "target_column": target_column
            }
            
            logger.info(f"Model training completed in {metrics['training_time_seconds']:.2f} seconds")
            return response_data, model_id
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise
    
    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a serialized model."""
        model_path = self.model_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)
    
    async def predict(self, model_id: str, features: Dict[str, float]) -> float:
        """Make prediction using a trained model."""
        try:
            model_artifacts = await self.load_model(model_id)
            sklearn_model = model_artifacts["sklearn_model"]
            feature_columns = model_artifacts["feature_columns"]
            
            # Validate features
            missing_features = set(feature_columns) - set(features.keys())
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Prepare feature vector
            feature_vector = np.array([[features[col] for col in feature_columns]])
            
            # Make prediction
            prediction = float(sklearn_model.predict(feature_vector)[0])
            logger.info(f"Prediction made for model {model_id}: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise


# ==================== API Router ====================
router = APIRouter(prefix="/api/v1", tags=["regression"])
model_manager = ModelManager()


@router.post("/train", response_model=TrainingResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train a regression model from uploaded CSV data.
    
    Args:
        file: CSV file upload
        target_column: Name of the target column
        test_size: Proportion of data for testing (0.0-1.0)
        random_state: Random seed
    
    Returns:
        Training metrics and model information
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type not allowed. Must be one of: {ALLOWED_EXTENSIONS}")
        
        # Read file content
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)} MB")
        
        # Parse CSV
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        # Validate test_size
        if not 0.0 < test_size < 1.0:
            raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
        
        # Train model
        response_data, model_id = await model_manager.train_model(
            data=data,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )
        
        return TrainingResponse(**response_data)
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/predict/{model_id}", response_model=PredictionResponse, responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
async def predict_endpoint(model_id: str, request: PredictionRequest):
    """
    Make a prediction using a trained model.
    
    Args:
        model_id: ID of the trained model
        request: Prediction request with features
    
    Returns:
        Prediction result
    """
    try:
        prediction = await model_manager.predict(model_id, request.features)
        
        return PredictionResponse(
            prediction=prediction,
            model_id=model_id,
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except ValueError as e:
        logger.error(f"Prediction validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/models/{model_id}", response_model=TrainingResponse, responses={404: {"model": ErrorResponse}})
async def get_model_info(model_id: str):
    """
    Get information about a trained model.
    
    Args:
        model_id: ID of the trained model
    
    Returns:
        Model information and metrics
    """
    try:
        model_artifacts = await model_manager.load_model(model_id)
        
        return TrainingResponse(
            model_id=model_id,
            timestamp=model_artifacts.get("training_date", datetime.now().isoformat()),
            metrics=model_artifacts["metrics"],
            coefficients=model_artifacts["coefficients"],
            p_values=model_artifacts["p_values"],
            diagnostics=model_artifacts["diagnostics"],
            model_path=str(self.model_dir / f"{model_id}.joblib"),
            feature_columns=model_artifacts["feature_columns"],
            target_column=model_artifacts["target_column"]
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Unexpected error retrieving model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_dir": str(MODEL_DIR),
        "models_available": len(list(MODEL_DIR.glob("*.joblib")))
    }


# ==================== FastAPI App ====================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting Regression API Service")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Available models: {len(list(MODEL_DIR.glob('*.joblib')))}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down Regression API Service")


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# Phase 1: Core Model Training and Serialization - iteration 3

# Phase 1: Core Model Training and Serialization - iteration 4

# Phase 1: Core Model Training and Serialization - iteration 5

# Phase 1: Core Model Training and Serialization - iteration 6

# Phase 1: Core Model Training and Serialization - iteration 7
