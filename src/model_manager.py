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
import statsmodels.api as sm
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError, validator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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


# ==================== Schemas ====================
class TrainingRequest(BaseModel):
    """Pydantic model for training request validation."""
    target_column: str = Field(..., min_length=1, description="Name of the target column")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    
    @validator("target_column")
    def validate_target_column(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()


class TrainingResponse(BaseModel):
    """Response model for training results."""
    model_id: str
    timestamp: str
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
    model_path: str


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    
    @validator("features")
    def validate_features(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float
    model_id: str
    timestamp: str


# ==================== Model Manager ====================
class ModelManager:
    """Manages regression model training, serialization, and prediction."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")
    
    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train a regression model with full diagnostics."""
        start_time = time.time()
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_state}"
        
        try:
            # Validate data
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            if data.empty:
                raise ValueError("Data is empty")
            
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)
            
            # Check for non-numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) != len(X.columns):
                raise ValueError("All features must be numeric after encoding")
            
            # Check for NaN values
            if X.isnull().any().any() or y.isnull().any():
                raise ValueError("Data contains NaN values")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train scikit-learn model
            sklearn_model = LinearRegression()
            sklearn_model.fit(X_train, y_train)
            
            # Train statsmodels for diagnostics
            X_train_sm = sm.add_constant(X_train)
            statsmodels_model = sm.OLS(y_train, X_train_sm).fit()
            
            # Make predictions
            y_pred_train = sklearn_model.predict(X_train)
            y_pred_test = sklearn_model.predict(X_test)
            
            # Calculate metrics
            r_squared = r2_score(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred_test)
            
            # Extract coefficients and p-values
            coefficients = {}
            p_values = {}
            
            # Get feature names from sklearn model
            feature_names = X.columns.tolist()
            
            # Map coefficients from statsmodels (which includes constant)
            sm_params = statsmodels_model.params
            sm_pvalues = statsmodels_model.pvalues
            
            # Handle constant term
            if "const" in sm_params.index:
                coefficients["intercept"] = float(sm_params["const"])
                p_values["intercept"] = float(sm_pvalues["const"])
            
            # Map feature coefficients
            for i, feature in enumerate(feature_names):
                if feature in sm_params.index:
                    coefficients[feature] = float(sm_params[feature])
                    p_values[feature] = float(sm_pvalues[feature])
                else:
                    # Fallback to sklearn coefficients
                    coefficients[feature] = float(sklearn_model.coef_[i])
                    p_values[feature] = float("nan")
            
            # Prepare model bundle
            model_bundle = {
                "sklearn_model": sklearn_model,
                "statsmodels_model": statsmodels_model,
                "feature_names": feature_names,
                "target_column": target_column,
                "model_id": model_id,
                "training_date": datetime.now().isoformat(),
                "metrics": {
                    "r_squared": float(r_squared),
                    "adjusted_r_squared": float(statsmodels_model.rsquared_adj),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "train_r_squared": float(r2_score(y_train, y_pred_train)),
                    "train_mse": float(mean_squared_error(y_train, y_pred_train)),
                    "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                    "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                    "training_time_seconds": float(time.time() - start_time)
                },
                "coefficients": coefficients,
                "p_values": p_values,
                "n_samples": int(len(data)),
                "n_features": int(len(feature_names))
            }
            
            # Serialize model
            model_path = self.model_dir / f"{model_id}.joblib"
            await asyncio.to_thread(joblib.dump, model_bundle, model_path)
            
            # Store in memory
            async with self._lock:
                self._models[model_id] = model_bundle
            
            logger.info(f"Model {model_id} trained successfully in {time.time() - start_time:.2f}s")
            
            return {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": model_bundle["metrics"],
                "coefficients": coefficients,
                "p_values": p_values,
                "r_squared": float(r_squared),
                "adjusted_r_squared": float(statsmodels_model.rsquared_adj),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "n_samples": int(len(data)),
                "n_features": int(len(feature_names)),
                "model_path": str(model_path)
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
    
    async def predict(self, model_id: str, features: Dict[str, float]) -> float:
        """Make prediction using a trained model."""
        try:
            # Get model from memory or load from disk
            async with self._lock:
                model_bundle = self._models.get(model_id)
            
            if model_bundle is None:
                model_path = self.model_dir / f"{model_id}.joblib"
                if not model_path.exists():
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                model_bundle = await asyncio.to_thread(joblib.load, model_path)
                async with self._lock:
                    self._models[model_id] = model_bundle
            
            # Prepare feature vector
            feature_names = model_bundle["feature_names"]
            feature_vector = []
            
            for feature in feature_names:
                if feature not in features:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing feature '{feature}' for prediction"
                    )
                feature_vector.append(features[feature])
            
            # Make prediction
            sklearn_model = model_bundle["sklearn_model"]
            prediction = await asyncio.to_thread(
                sklearn_model.predict, [feature_vector]
            )
            
            logger.info(f"Prediction made with model {model_id}")
            return float(prediction[0])
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        for model_file in self.model_dir.glob("*.joblib"):
            try:
                model_bundle = await asyncio.to_thread(joblib.load, model_file)
                models.append({
                    "model_id": model_bundle["model_id"],
                    "training_date": model_bundle["training_date"],
                    "metrics": model_bundle["metrics"],
                    "n_features": model_bundle["n_features"]
                })
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {str(e)}")
        return models
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        model_path = self.model_dir / f"{model_id}.joblib"
        
        async with self._lock:
            self._models.pop(model_id, None)
        
        if model_path.exists():
            await asyncio.to_thread(model_path.unlink)
            logger.info(f"Model {model_id} deleted")
            return True
        else:
            logger.warning(f"Model {model_id} not found for deletion")
            return False


# ==================== API ====================
app = FastAPI(
    title="Regression API Service",
    description="Production-grade service for training and serving regression models",
    version="1.0.0"
)

model_manager = ModelManager()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "regression-api-service"
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(
    file: UploadFile = File(..., description="CSV file with training data"),
    target_column: str = File(..., description="Name of the target column"),
    test_size: float = File(0.2, description="Test set proportion (0.1-0.5)"),
    random_state: int = File(42, description="Random seed")
) -> TrainingResponse:
    """Train a regression model from CSV data."""
    try:
        # Validate parameters
        request_data = TrainingRequest(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )
        
        # Read CSV data
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Parse CSV with error handling
        try:
            data = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        
        if data.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        # Train model
        result = await model_manager.train_model(
            data=data,
            target_column=request_data.target_column,
            test_size=request_data.test_size,
            random_state=request_data.random_state
        )
        
        return TrainingResponse(**result)
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict(
    model_id: str,
    request: PredictionRequest
) -> PredictionResponse:
    """Make prediction using a trained model."""
    try:
        prediction = await model_manager.predict(model_id, request.features)
        
        return PredictionResponse(
            prediction=prediction,
            model_id=model_id,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List all trained models."""
    try:
        models = await model_manager.list_models()
        return {
            "count": len(models),
            "models": models,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.delete("/models/{model_id}")
async def delete_model(model_id: str) -> Dict[str, Any]:
    """Delete a trained model."""
    try:
        deleted = await model_manager.delete_model(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return {
            "status": "deleted",
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Regression API Service starting up")
    # Ensure model directory exists
    Path("models").mkdir(exist_ok=True)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Regression API Service shutting down")


# ==================== Main entry point ====================
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
