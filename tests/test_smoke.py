import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.main import app

client = TestClient(app)

def test_health_check():
    """Verify health endpoint returns healthy status with model info"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data

def test_predict_endpoint():
    """Verify predict endpoint works with valid customer data"""
    customer_data = {
        "customerID": "string",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "Yes",
        "tenure": 72,
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 150,
        "TotalCharges": 10000,
        "PhoneService": "Yes",
        "MultipleLines": "No phone service",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No"
    }
    
    response = client.post("/predict", json=customer_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "customer_id" in data
    assert "churn_risk" in data
    assert "validated_data" in data

def test_predict_invalid_data():
    """Verify predict endpoint handles invalid data properly"""
    invalid_data = {"wrong_field": "test"}
    
    response = client.post("/predict", json=invalid_data)
    
    assert response.status_code == 422  # validation error

def test_api_connectivity():
    """Quick connectivity smoke test"""
    response = client.get("/health")
    assert response.status_code == 200