from fastapi import FastAPI
from mlp_package import LoadModel
from pathlib import Path
import time

# Prometheus
from prometheus_fastapi_instrumentator import Instrumentator

#local modules
from .schemas.predict import CustomerChurnBase
from .schemas.health import HealthResponse
from .middleware.monitoring import APILoggingMiddleware
from .logging_config import logger
from .metrics import (
    TOTAL_PREDICTIONS,
    TOTAL_ERRORS,
    PREDICTION_LATENCY,
    MODEL_LOADED,
    CURRENT_THRESHOLD
)


#============================================================================
# LOADING MODEL
#============================================================================
# Get the absolute path to the directory containing this file
CURRENT_DIR = Path(__file__).parent.absolute()
# Go up two levels to the project root (adjust based on your structure)
PROJECT_ROOT = CURRENT_DIR.parent.parent
# Build absolute paths
model_path = PROJECT_ROOT / 'artifacts' / 'models' / 'model_weights_v1.pth'
pipeline_path = PROJECT_ROOT / 'artifacts' / 'models' / 'data_processing.joblib'

if not model_path.exists():
    logger.warning(f"model not found in path{model_path}")
if not pipeline_path.exists():
    logger.warning(f"data transformation pipeline not found in path{pipeline_path}")

model = LoadModel(str(model_path), str(pipeline_path))

#=============================================================================
# FASTAPI
#=============================================================================
app = FastAPI()

app.add_middleware(
    APILoggingMiddleware,
    trace_id_header="X-Trace-ID",
    exclude_paths=["/health", "/docs"],
    log_request_body=False,
    log_response_body=False,
)

#=============================================================================
# ENDPOINTS
#=============================================================================
@app.post("/predict")
async def predict_churn(customer: CustomerChurnBase):
    """
    Accepts customer data and returns churn prediction
    """
    
    predictions = model.predict([customer.model_dump()])

    return {
        "customer_id": "generated_id",
        "churn_risk": "High",  # Example output
        "validated_data": predictions
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check for MLP model"""
    
    # Check if model is loaded
    model_loaded = model is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }