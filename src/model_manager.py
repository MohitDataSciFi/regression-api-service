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
        logging.StreamHandler(),
        logging.FileHandler('regression_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.csv', '.txt'}


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature columns. If None, all other columns used")

    @validator('target_column')
    def validate_target_column(cls, v):
        if not v or not v.strip():
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
    timestamp: str
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_p_value: float
    mse: float
    rmse: float
    mae: float
    n_samples: int
    n_features: int
    model_path: str


class ModelManager:
    """Manages regression model training, diagnostics, and serialization."""

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
                if feature_columns:
                    X = data[feature_columns].copy()
                else:
                    X = data.drop(columns=[target_column]).copy()
                
                y = data[target_column].copy()

                # Handle categorical variables
                X = self._encode_categorical(X)

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

                # Make predictions
                y_pred_train = sklearn_model.predict(X_train)
                y_pred_test = sklearn_model.predict(X_test)

                # Compute metrics
                metrics = self._compute_metrics(y_test, y_pred_test, y_train, y_pred_train)

                # Extract coefficients and p-values
                coefficients = self._extract_coefficients(sklearn_model, X.columns)
                p_values = self._extract_p_values(statsmodels_model, X.columns)

                # Create model ID and save
                model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
                model_path = self.model_dir / f"{model_id}.joblib"

                # Save both models and metadata
                model_data = {
                    'sklearn_model': sklearn_model,
                    'statsmodels_model': statsmodels_model,
                    'feature_names': list(X.columns),
                    'target_name': target_column,
                    'metadata': {
                        'model_id': model_id,
                        'timestamp': datetime.now().isoformat(),
                        'test_size': test_size,
                        'random_state': random_state,
                        'n_samples': len(data),
                        'n_features': len(X.columns)
                    }
                }

                joblib.dump(model_data, model_path)

                training_time = time.time() - start_time
                logger.info(f"Model training completed in {training_time:.2f}s. Model ID: {model_id}")

                return TrainingResponse(
                    model_id=model_id,
                    timestamp=datetime.now().isoformat(),
                    metrics=metrics,
                    coefficients=coefficients,
                    p_values=p_values,
                    r_squared=float(statsmodels_model.rsquared),
                    adjusted_r_squared=float(statsmodels_model.rsquared_adj),
                    f_statistic=float(statsmodels_model.fvalue),
                    f_p_value=float(statsmodels_model.f_pvalue),
                    mse=metrics['mse'],
                    rmse=metrics['rmse'],
                    mae=metrics['mae'],
                    n_samples=len(data),
                    n_features=len(X.columns),
                    model_path=str(model_path)
                )

            except Exception as e:
                logger.error(f"Model training failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    def _validate_data(self, data: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]]):
        """Validate input data."""
        if data.empty:
            raise ValueError("Data is empty")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        if feature_columns:
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
        
        # Check for non-numeric target
        if not pd.api.types.is_numeric_dtype(data[target_column]):
            raise ValueError("Target column must be numeric")
        
        # Check for NaN in target
        if data[target_column].isna().any():
            raise ValueError("Target column contains NaN values")

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using one-hot encoding."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        return X

    def _compute_metrics(
        self, y_test: np.ndarray, y_pred_test: np.ndarray,
        y_train: np.ndarray, y_pred_train: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive regression metrics."""
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        # Training metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'train_mse': float(train_mse),
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'train_r2': float(train_r2)
        }

    def _extract_coefficients(self, model: LinearRegression, feature_names: List[str]) -> Dict[str, float]:
        """Extract model coefficients."""
        coefficients = {}
        for name, coef in zip(feature_names, model.coef_):
            coefficients[name] = float(coef)
        return coefficients

    def _extract_p_values(self, model: OLS, feature_names: List[str]) -> Dict[str, float]:
        """Extract p-values from statsmodels model."""
        p_values = {}
        # First value is for the constant
        for i, name in enumerate(['const'] + list(feature_names)):
            if i < len(model.pvalues):
                p_values[name] = float(model.pvalues[i])
        return p_values


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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


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
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File extension {file_extension} not allowed. Use {ALLOWED_EXTENSIONS}")

        # Read and validate file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE} bytes")

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
            raise HTTPException(status_code=422, detail=e.errors())

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
async def get_model_info(model_id: str):
    """Get information about a trained model."""
    model_path = MODEL_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    try:
        model_data = joblib.load(model_path)
        return {
            "model_id": model_id,
            "metadata": model_data['metadata'],
            "feature_names": model_data['feature_names'],
            "target_name": model_data['target_name']
        }
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/models")
async def list_models():
    """List all trained models."""
    models = []
    for model_file in MODEL_DIR.glob("*.joblib"):
        try:
            model_data = joblib.load(model_file)
            models.append({
                "model_id": model_data['metadata']['model_id'],
                "timestamp": model_data['metadata']['timestamp'],
                "n_samples": model_data['metadata']['n_samples'],
                "n_features": model_data['metadata']['n_features']
            })
        except Exception as e:
            logger.warning(f"Failed to load model metadata from {model_file}: {str(e)}")
    
    return {"models": models, "count": len(models)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")