from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from typing import Optional
import time
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize NLTK (matches your Streamlit app)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# === Configuration ===
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
API_VERSION = "2.1.0"  # Updated version

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === FastAPI Setup ===
app = FastAPI(
    title="Job Scam Detection API",
    description="API for detecting fraudulent job postings using ML",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url=None
)

# CORS (Keep your existing config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# === Model & Utilities ===
def clean_text(text: str) -> str:
    """Identical to Streamlit app's cleaning function"""
    text = str(text).lower()
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

try:
    logger.info(f"Loading model from {MODEL_PATH}...")
    start_time = time.time()
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
except Exception as e:
    logger.critical(f"Model load failed: {str(e)}")
    raise RuntimeError("Model loading failed - check logs")

# === Schemas ===
class JobRequest(BaseModel):
    description: str
    threshold: Optional[float] = 0.5
    raw_text: Optional[bool] = False  # New: Skip cleaning if True

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_version: str
    processing_time_ms: float
    risk_level: str  # New: "low", "medium", "high"

# === Core Endpoints ===
@app.post("/predict", response_model=PredictionResponse)
async def predict(job: JobRequest, request: Request):
    """
    Predict fraud probability with enhanced features:
    - Same text cleaning as Streamlit app
    - Risk level categorization
    - Optional raw text processing
    """
    start_time = time.time()
    
    try:
        # Input validation
        if not job.description.strip():
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        
        if len(job.description) < 20:
            raise HTTPException(status_code=400, detail="Description too short (min 20 chars)")
            
        if not 0 <= job.threshold <= 1:
            raise HTTPException(status_code=400, detail="Threshold must be 0-1")
        
        # Preprocessing
        text_to_predict = job.description if job.raw_text else clean_text(job.description)
        
        # Prediction
        proba = model.predict_proba([text_to_predict])[0][1]
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Risk categorization (matches Streamlit)
        if proba > 0.7:
            risk_level = "high"
        elif proba > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        logger.info(f"Prediction completed in {processing_time_ms:.1f}ms | Risk: {risk_level}")
        
        return {
            "fraud_probability": float(proba),
            "is_fraud": proba > job.threshold,
            "model_version": API_VERSION,
            "processing_time_ms": processing_time_ms,
            "risk_level": risk_level
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction error")

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": API_VERSION,
        "model_timestamp": os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    }

# === Batch Endpoint (New) ===
class BatchRequest(BaseModel):
    descriptions: list[str]
    threshold: Optional[float] = 0.5

@app.post("/batch_predict")
async def batch_predict(batch: BatchRequest):
    """Process multiple job descriptions at once"""
    start_time = time.time()
    
    try:
        if len(batch.descriptions) > 100:
            raise HTTPException(status_code=400, detail="Max 100 descriptions per batch")
            
        cleaned_texts = [clean_text(desc) for desc in batch.descriptions]
        probabilities = model.predict_proba(cleaned_texts)[:, 1]
        
        return {
            "results": [
                {
                    "description": desc,
                    "fraud_probability": float(prob),
                    "is_fraud": prob > batch.threshold
                }
                for desc, prob in zip(batch.descriptions, probabilities)
            ],
            "processing_time_ms": (time.time() - start_time) * 1000
        }
    except Exception as e:
        logger.error(f"Batch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch processing error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )