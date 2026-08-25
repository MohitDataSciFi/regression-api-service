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

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('regression_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_EXTENSIONS = {'.csv', '.txt'}


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns. If None, all other columns used")

    @validator('target_column')
    def validate_target_column(cls, v):
        if not v.strip():
            raise ValueError('Target column cannot be empty')
        return v.strip()

    @validator('feature_columns')
    def validate_feature_columns(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError('Feature columns list cannot be empty')
            if len(v) != len(set(v)):
                raise ValueError('Feature columns must be unique')
        return v


class TrainingResponse(BaseModel):
    """Response model for training results."""
    model_id: str
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    feature_importance: Dict[str, float]
    training_timestamp: str
    model_path: str
    data_shape: Dict[str, int]


class ModelManager:
    """Manages regression model training, serialization, and diagnostics."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with directory: {self.model_dir}")

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_state: int,
        feature_columns: Optional[List[str]] = None
    ) -> TrainingResponse:
        """Train an OLS regression model with comprehensive diagnostics."""
        
        async with self._lock:
            try:
                logger.info(f"Starting model training with target: {target_column}")
                start_time = time.time()

                # Validate data
                if data.empty:
                    raise ValueError("Data cannot be empty")
                
                if target_column not in data.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                
                # Prepare features
                if feature_columns is None:
                    feature_columns = [col for col in data.columns if col != target_column]
                else:
                    missing_cols = set(feature_columns) - set(data.columns)
                    if missing_cols:
                        raise ValueError(f"Missing feature columns: {missing_cols}")
                
                if not feature_columns:
                    raise ValueError("No feature columns available for training")
                
                # Extract features and target
                X = data[feature_columns].copy()
                y = data[target_column].copy()
                
                # Handle missing values
                if X.isnull().any().any() or y.isnull().any():
                    logger.warning("Missing values detected, dropping rows")
                    mask = ~(X.isnull().any(axis=1) | y.isnull())
                    X = X[mask]
                    y = y[mask]
                
                # Convert categorical variables to dummy variables
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    logger.info(f"Converting categorical columns to dummies: {list(categorical_cols)}")
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                
                # Ensure numeric data
                X = X.apply(pd.to_numeric, errors='coerce')
                y = pd.to_numeric(y, errors='coerce')
                
                # Drop any rows with NaN after conversion
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
                
                if len(X) < 10:
                    raise ValueError("Insufficient data after cleaning. Need at least 10 samples")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train scikit-learn model
                sklearn_model = LinearRegression()
                sklearn_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = sklearn_model.predict(X_train)
                y_pred_test = sklearn_model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'r2_train': float(r2_score(y_train, y_pred_train)),
                    'r2_test': float(r2_score(y_test, y_pred_test)),
                    'mse_train': float(mean_squared_error(y_train, y_pred_train)),
                    'mse_test': float(mean_squared_error(y_test, y_pred_test)),
                    'rmse_train': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                    'rmse_test': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                    'mae_train': float(mean_absolute_error(y_train, y_pred_train)),
                    'mae_test': float(mean_absolute_error(y_test, y_pred_test)),
                    'training_time': float(time.time() - start_time)
                }
                
                # Train statsmodels for diagnostics
                X_const = add_constant(X)
                stats_model = OLS(y, X_const).fit()
                
                # Extract diagnostics
                diagnostics = {
                    'coefficients': stats_model.params.to_dict(),
                    'p_values': stats_model.pvalues.to_dict(),
                    'std_errors': stats_model.bse.to_dict(),
                    'conf_int_lower': stats_model.conf_int()[0].to_dict(),
                    'conf_int_upper': stats_model.conf_int()[1].to_dict(),
                    'aic': float(stats_model.aic),
                    'bic': float(stats_model.bic),
                    'f_statistic': float(stats_model.fvalue),
                    'f_p_value': float(stats_model.f_pvalue),
                    'durbin_watson': float(durbin_watson(stats_model.resid)),
                    'condition_number': float(np.linalg.cond(X_const))
                }
                
                # Breusch-Pagan test for heteroscedasticity
                try:
                    bp_test = het_breuschpagan(stats_model.resid, X_const)
                    diagnostics['breusch_pagan'] = {
                        'lm_statistic': float(bp_test[0]),
                        'lm_p_value': float(bp_test[1]),
                        'f_statistic': float(bp_test[2]),
                        'f_p_value': float(bp_test[3])
                    }
                except Exception as e:
                    logger.warning(f"Breusch-Pagan test failed: {e}")
                    diagnostics['breusch_pagan'] = None
                
                # Feature importance (absolute coefficients)
                feature_importance = {
                    col: abs(float(coef)) 
                    for col, coef in zip(X.columns, sklearn_model.coef_)
                }
                
                # Sort by importance
                feature_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
                
                # Generate model ID
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
                model_path = self.model_dir / f"{model_id}.joblib"
                
                # Save model and metadata
                model_data = {
                    'sklearn_model': sklearn_model,
                    'statsmodels_model': stats_model,
                    'feature_columns': list(X.columns),
                    'target_column': target_column,
                    'metrics': metrics,
                    'diagnostics': diagnostics,
                    'feature_importance': feature_importance,
                    'training_timestamp': datetime.now().isoformat(),
                    'model_id': model_id
                }
                
                joblib.dump(model_data, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Create response
                response = TrainingResponse(
                    model_id=model_id,
                    metrics=metrics,
                    diagnostics=diagnostics,
                    feature_importance=feature_importance,
                    training_timestamp=datetime.now().isoformat(),
                    model_path=str(model_path),
                    data_shape={
                        'original_samples': int(len(data)),
                        'cleaned_samples': int(len(X)),
                        'features': int(X.shape[1]),
                        'train_samples': int(len(X_train)),
                        'test_samples': int(len(X_test))
                    }
                )
                
                logger.info(f"Training completed in {metrics['training_time']:.2f} seconds")
                return response
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Initialize FastAPI app and model manager
app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0"
)
model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/train", response_model=TrainingResponse)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file containing training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed"),
    feature_columns: Optional[str] = File(None, description="Comma-separated feature columns")
) -> TrainingResponse:
    """Train a regression model from uploaded CSV data."""
    
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
            )
        
        # Read and validate file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
            )
        
        # Parse CSV
        try:
            data = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        # Parse feature columns if provided
        feature_list = None
        if feature_columns:
            feature_list = [col.strip() for col in feature_columns.split(',') if col.strip()]
        
        # Validate request parameters
        try:
            request = TrainingRequest(
                target_column=target_column,
                test_size=test_size,
                random_state=random_state,
                feature_columns=feature_list
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        # Train model
        response = await model_manager.train_model(
            data=data,
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
        
        model_data = joblib.load(model_path)
        return {
            "model_id": model_data['model_id'],
            "feature_columns": model_data['feature_columns'],
            "target_column": model_data['target_column'],
            "metrics": model_data['metrics'],
            "diagnostics": model_data['diagnostics'],
            "feature_importance": model_data['feature_importance'],
            "training_timestamp": model_data['training_timestamp']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List all trained models."""
    try:
        model_files = list(MODEL_DIR.glob("*.joblib"))
        model_ids = [f.stem for f in model_files]
        return {"models": model_ids}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")