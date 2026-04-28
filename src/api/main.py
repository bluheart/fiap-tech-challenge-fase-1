from fastapi import FastAPI
from .schemas.predict import CustomerChurnBase
from .schemas.health import HealthResponse
from .middleware.monitoring import MonitoringMiddleware
from mlp_package import LoadModel
from pathlib import Path
import time

# Get the absolute path to the directory containing this file
CURRENT_DIR = Path(__file__).parent.absolute()
# Go up two levels to the project root (adjust based on your structure)
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Build absolute paths
model_path = PROJECT_ROOT / 'artifacts' / 'models' / 'model_weights_v1.pth'
pipeline_path = PROJECT_ROOT / 'artifacts' / 'models' / 'data_processing.joblib'

app = FastAPI()

# Check if files exist and load model
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")
if not pipeline_path.exists():
    raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")

model = LoadModel(str(model_path), str(pipeline_path))

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