from fastapi import FastAPI, Request,  HTTPException
from mlp_package import LoadModel
from pathlib import Path
from typing import List
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
    PREDICTION_LATENCY,
    TOTAL_ERRORS,
    MODEL_LOADED,
    CURRENT_THRESHOLD
)

#============================================================================
# DEFINING THRESHOLD
#============================================================================
# change the threshold based on the balance between recall and precision
THRESHOLD = 0.20

#============================================================================
# LOADING MODEL
#============================================================================
# Get the absolute path to the directory containing this file
CURRENT_DIR = Path(__file__).parent.absolute()
# Go up two levels to the project root (adjust based on your structure)
PROJECT_ROOT = CURRENT_DIR.parent.parent
# Build absolute paths
model_path = PROJECT_ROOT / 'src' / 'models' / 'artifacts' / 'model_weights_v1.pth'
pipeline_path = PROJECT_ROOT / 'src' / 'models' / 'artifacts' / 'data_processing.joblib'

if not model_path.exists():
    logger.warning("model not found", extra={"path": str(model_path)})
if not pipeline_path.exists():
    logger.warning("data transformation pipeline not found" , extra={"path":str(pipeline_path)})

model = LoadModel(str(model_path), str(pipeline_path))
if model:
    logger.info("model_loaded", extra={"path": str(model_path)})
else:
    logger.error("model_not_loaded", extra={"path": str(model_path)})

# Prometheus metrics
MODEL_LOADED.set(1 if model else 0)
CURRENT_THRESHOLD.set(THRESHOLD)
    

#=============================================================================
# FASTAPI APP
#=============================================================================
app = FastAPI(
    title="API MLP",
    description="API for telecom customer churn classification",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    APILoggingMiddleware,
    trace_id_header="X-Trace-ID",
    exclude_paths=["/health", "/docs", "/redoc"],
    log_request_body=False,
    log_response_body=False,
)

# ============================================================================
# Prometheus metrics + endpoint
# ============================================================================
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

#=============================================================================
# ENDPOINTS
#=============================================================================
@app.post("/predict")
async def predict_churn(customers: List[CustomerChurnBase], request: Request):
    """
    Predict customer churn probability for multiple customers.
    
    Uses the pre-loaded machine learning model to calculate churn probabilities
    for a batch of customers and returns binary predictions based on a configured threshold.
    
    Args:
        customers: List of validated customer data conforming to CustomerChurnBase schema
        request: FastAPI request object for accessing request state (trace_id)
    
    Returns:
        List of dicts, each containing:
            - customer_id: The customer's identifier
            - churn: Binary prediction ("yes"/"no") based on threshold comparison
            - threshold_used: The probability threshold used for classification
            - probs: Probability scores for both classes (churn/no churn)
    
    Raises:
        HTTPException 503: If the model has not been loaded or initialized
        HTTPException 400: If the customers list is empty
    """
    if not model:
        MODEL_LOADED.set(0)  # Update metric
        raise HTTPException(status_code=503, detail="Model not available")
    
    if not customers:
        raise HTTPException(status_code=400, detail="Customers list cannot be empty")
    
    trace_id = getattr(request.state, 'trace_id', 'N/A')
    customers_dicts = [customer.model_dump() for customer in customers]
    
    start = time.perf_counter()
    
    try:
        # Get predictions for all customers at once
        predictions = model.predict(customers_dicts)
        
        if not predictions or len(predictions) == 0:
            TOTAL_ERRORS.labels(error_type="empty_prediction").inc()
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Process results for each customer
        results = []
        for i, customer_dict in enumerate(customers_dicts):
            prediction_result = {
                "customer_id": customer_dict['customerID'],
                "churn": "yes" if predictions[i][1] > THRESHOLD else "no",
                "threshold_used": THRESHOLD,
                "probs": {
                    "yes": predictions[i][1],
                    "no": predictions[i][0]
                }
            }
            results.append(prediction_result)
        
        latency = time.perf_counter() - start
        
        # Record metrics - increment by number of predictions
        TOTAL_PREDICTIONS.labels(threshold=str(THRESHOLD)).inc(len(customers))
        PREDICTION_LATENCY.observe(latency)
        
        logger.info("batch prediction completed", extra={
            "trace_id": trace_id,
            "batch_size": len(customers),
            "threshold": THRESHOLD,
            "latency": latency
        })
        
        return results
        
    except Exception as e:
        TOTAL_ERRORS.labels(error_type="prediction_error").inc()
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")
        

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and model availability.
    
    Performs a lightweight check to determine if the ML model has been 
    properly loaded and the service is ready to handle prediction requests.
    
    Returns:
        HealthResponse containing:
            - status: "healthy" if model is loaded, "unhealthy" otherwise
            - model_loaded: Boolean indicating whether the model is available
            - timestamp: Unix timestamp of the health check
    
    Note:
        This endpoint is designed for use with orchestration systems (e.g., 
        Kubernetes liveness/readiness probes) and monitoring services.
    
    Example Response:
        {
            "status": "healthy",
            "model_loaded": true,
            "timestamp": 1714567890.123
        }
    """
    # Check if model is loaded
    model_loaded = model is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }