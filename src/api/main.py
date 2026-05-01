from fastapi import FastAPI, Request,  HTTPException
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
    PREDICTION_LATENCY,
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
model_path = PROJECT_ROOT / 'artifacts' / 'models' / 'model_weights_v1.pth'
pipeline_path = PROJECT_ROOT / 'artifacts' / 'models' / 'data_processing.joblib'

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
async def predict_churn(customer: CustomerChurnBase, request: Request):
    """
    Accepts customer data and returns churn prediction
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modelo not available")

    trace_id = getattr(request.state, 'trace_id', 'N/A')
    customer_dict = customer.model_dump()

    # latency of prediction
    start = time.perf_counter()

    predictions = model.predict([customer_dict])

    latency = time.perf_counter() - start

    # Prometheus metrics
    TOTAL_PREDICTIONS.labels(threshold=THRESHOLD).inc()
    PREDICTION_LATENCY.observe(latency)

    #log
    logger.info("prediction completed", extra={
        "trace_id": trace_id,
        "customer_id": customer_dict['customerID'],
        "churn_prob": predictions[0][1],
        "no_churn_prob": predictions[0][0],
        "threshold": THRESHOLD
    })

    return {
        "customer_id": customer_dict['customerID'],
        "churn": "yes" if predictions[0][1]>THRESHOLD else "no",
        "threshold_used": THRESHOLD,
        "probs": {
            "yes":  predictions[0][1],
            "no": predictions[0][0]
        }
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