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


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


class ModelValidationError(Exception):
    """Custom exception for model validation errors."""
    pass


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns to use")

    @validator("target_column")
    def validate_target_column(cls, v):
        if not v or not v.strip():
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
    sample_size: int
    feature_count: int
    model_path: str
    diagnostics: Dict[str, Any]


class ModelManager:
    """Manages regression model training, serialization, and diagnostics."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        self._models: Dict[str, Dict[str, Any]] = {}

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train a regression model with full diagnostics."""
        async with self._lock:
            try:
                # Validate data
                if data.empty:
                    raise ModelValidationError("DataFrame is empty")
                
                if target_column not in data.columns:
                    raise ModelValidationError(f"Target column '{target_column}' not found in data")
                
                # Prepare features
                if feature_columns is None:
                    feature_columns = [col for col in data.columns if col != target_column]
                else:
                    # Validate feature columns exist
                    missing_cols = set(feature_columns) - set(data.columns)
                    if missing_cols:
                        raise ModelValidationError(f"Missing feature columns: {missing_cols}")
                
                if not feature_columns:
                    raise ModelValidationError("No feature columns available for training")
                
                # Extract features and target
                X = data[feature_columns].copy()
                y = data[target_column].copy()
                
                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    logger.warning("Missing values detected, dropping rows with NaN")
                    mask = X.notna().all(axis=1) & y.notna()
                    X = X[mask]
                    y = y[mask]
                
                # Convert to numeric
                X = X.apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(y, errors='coerce')
                
                # Drop any remaining NaN after conversion
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y = y[mask]
                
                if len(X) < 10:
                    raise ModelValidationError("Insufficient data after cleaning (minimum 10 samples required)")
                
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
                
                # Make predictions
                y_pred = sklearn_model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Get coefficients and p-values
                coefficients = dict(zip(feature_columns, sklearn_model.coef_))
                coefficients['intercept'] = sklearn_model.intercept_
                
                p_values = {}
                for i, col in enumerate(feature_columns):
                    p_values[col] = float(stats_model.pvalues[i + 1])  # +1 for constant
                p_values['intercept'] = float(stats_model.pvalues[0])
                
                # Calculate adjusted R²
                n = len(X_test)
                p = len(feature_columns)
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                
                # Diagnostics
                diagnostics = self._compute_diagnostics(stats_model, X_test, y_test, y_pred)
                
                # Generate model ID
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
                
                # Serialize model
                model_path = self.model_dir / f"{model_id}.joblib"
                model_data = {
                    'sklearn_model': sklearn_model,
                    'statsmodels_model': stats_model,
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'training_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'test_size': test_size,
                        'random_state': random_state,
                        'sample_size': len(X),
                        'feature_count': len(feature_columns)
                    }
                }
                
                # Save model
                joblib.dump(model_data, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Store in memory
                self._models[model_id] = {
                    'path': str(model_path),
                    'metadata': model_data['training_metadata']
                }
                
                # Prepare response
                response = {
                    'model_id': model_id,
                    'training_timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'r2': float(r2),
                        'adjusted_r2': float(adjusted_r2),
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'mae': float(mae)
                    },
                    'coefficients': coefficients,
                    'p_values': p_values,
                    'r_squared': float(r2),
                    'adjusted_r_squared': float(adjusted_r2),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'sample_size': len(X),
                    'feature_count': len(feature_columns),
                    'model_path': str(model_path),
                    'diagnostics': diagnostics
                }
                
                logger.info(f"Model training completed: {model_id}")
                return response
                
            except ModelValidationError as e:
                logger.error(f"Validation error: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Training error: {str(e)}", exc_info=True)
                raise ModelTrainingError(f"Model training failed: {str(e)}")

    def _compute_diagnostics(
        self,
        stats_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute additional statistical diagnostics."""
        try:
            # Residuals
            residuals = y_test - y_pred
            
            # Durbin-Watson statistic
            dw_stat = durbin_watson(residuals)
            
            # Breusch-Pagan test for heteroscedasticity
            X_test_const = add_constant(X_test)
            bp_test = het_breuschpagan(residuals, X_test_const)
            
            # Shapiro-Wilk test for normality of residuals
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            
            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(residuals)
            
            # F-statistic and p-value
            f_stat = float(stats_model.fvalue)
            f_pvalue = float(stats_model.f_pvalue)
            
            # AIC and BIC
            aic = float(stats_model.aic)
            bic = float(stats_model.bic)
            
            return {
                'durbin_watson': float(dw_stat),
                'breusch_pagan': {
                    'lm_statistic': float(bp_test[0]),
                    'lm_pvalue': float(bp_test[1]),
                    'f_statistic': float(bp_test[2]),
                    'f_pvalue': float(bp_test[3])
                },
                'shapiro_wilk': {
                    'statistic': float(shapiro_stat),
                    'pvalue': float(shapiro_p)
                },
                'jarque_bera': {
                    'statistic': float(jb_stat),
                    'pvalue': float(jb_p)
                },
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'aic': aic,
                'bic': bic,
                'residual_stats': {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'min': float(np.min(residuals)),
                    'max': float(np.max(residuals))
                }
            }
        except Exception as e:
            logger.warning(f"Could not compute all diagnostics: {str(e)}")
            return {'error': str(e)}

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a trained model from disk."""
        async with self._lock:
            try:
                model_path = self.model_dir / f"{model_id}.joblib"
                if not model_path.exists():
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                
                model_data = joblib.load(model_path)
                logger.info(f"Model {model_id} loaded successfully")
                return model_data
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    async def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        model_data = await self.load_model(model_id)
        
        try:
            sklearn_model = model_data['sklearn_model']
            feature_columns = model_data['feature_columns']
            
            # Validate features
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {missing_cols}"
                )
            
            # Prepare features
            X = data[feature_columns].copy()
            X = X.apply(pd.to_numeric, errors='coerce')
            
            # Handle NaN
            if X.isnull().any().any():
                raise HTTPException(
                    status_code=400,
                    detail="Input data contains NaN values"
                )
            
            # Make predictions
            predictions = sklearn_model.predict(X)
            logger.info(f"Predictions made for model {model_id}")
            return predictions
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction error for model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Initialize FastAPI app and model manager
app = FastAPI(
    title="Regression API Service",
    description="Production-grade API for training and serving regression models",
    version="1.0.0"
)

model_manager = ModelManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_count": len(model_manager._models)
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(
    file: UploadFile = File(...),
    target_column: str = File(...),
    test_size: float = File(0.2),
    random_state: int = File(42),
    feature_columns: Optional[str] = File(None)
):
    """
    Train a regression model from uploaded CSV data.
    
    Args:
        file: CSV file containing training data
        target_column: Name of the target column
        test_size: Proportion of data to use for testing (0.1-0.5)
        random_state: Random seed for reproducibility
        feature_columns: Comma-separated list of feature columns (optional)
    
    Returns:
        TrainingResponse with model metrics and diagnostics
    """
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
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
            )
        
        # Parse CSV
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        
        # Parse feature columns if provided
        feature_list = None
        if feature_columns:
            feature_list = [col.strip() for col in feature_columns.split(",") if col.strip()]
        
        # Validate request parameters
        try:
            request = TrainingRequest(
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                feature_columns=feature_list
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        
        # Train model
        result = await model_manager.train_model(
            data=data,
            target_column=request.target_column,
            test_size=request.test_size,
            random_state=request.random_state,
            feature_columns=request.feature_columns
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/{model_id}")
async def predict(
    model_id: str,
    file: UploadFile = File(...)
):
    """
    Make predictions using a trained model.
    
    Args:
        model_id: ID of the trained model
        file: CSV file containing features for prediction
    
    Returns:
        Dictionary with predictions
    """
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
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
            )
        
        # Parse CSV
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        
        # Make predictions
        predictions = await model_manager.predict(model_id, data)
        
        return {
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a trained model."""
    try:
        model_data = await model_manager.load_model(model_id)
        metadata = model_data['training_metadata']
        
        return {
            "model_id": model_id,
            "metadata": metadata,
            "feature_columns": model_data['feature_columns'],
            "target_column": model_data['target_column']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/models")
async def list_models():
    """List all trained models."""
    try:
        models = []
        for model_id, info in model_manager._models.items():
            models.append({
                "model_id": model_id,
                "path": info['path'],
                "metadata": info['metadata']
            })
        return {"models": models, "count": len(models)}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Phase 1: Core Model Training and Serialization - iteration 3
