from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import os
from typing import Dict, Any
import pandas as pd
import logging
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="House Price Prediction API")

# Define paths
MODEL_PATH = os.path.join('models', 'model.joblib')
PREPROCESSOR_PATH = os.path.join('data', 'processed', 'preprocessor.joblib')

# Load model and preprocessor
try:
    logger.info("Loading model and preprocessor...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {str(e)}")
    model = None
    preprocessor = None


class HouseFeatures(BaseModel):
    features: Dict[str, Any]


@app.post("/predict")
async def predict(house_data: HouseFeatures):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")

    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([house_data.features])
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input columns: {df.columns.tolist()}")

        # Ensure all required columns are present
        try:
            # Preprocess input data
            X = preprocessor.transform(df)
            logger.info(f"Preprocessed data shape: {X.shape}")

            dtest = xgb.DMatrix(X)

            # Make prediction
            prediction = model.predict(dtest)
            logger.info(f"Raw prediction: {prediction}")

            return {"predicted_price": float(prediction[0])}

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Error processing input data: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing input data: {str(e)}"
        )


@app.get("/health")
async def health_check():
    if model is None or preprocessor is None:
        return {
            "status": "unhealthy",
            "detail": "Model or preprocessor not loaded"
        }
    return {"status": "healthy"}


# Optional: Add endpoint to get required features
@app.get("/features")
async def get_required_features():
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded")
    try:
        # Get feature names from preprocessor
        feature_names = preprocessor.get_feature_names_out()
        return {"required_features": feature_names.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature names: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)