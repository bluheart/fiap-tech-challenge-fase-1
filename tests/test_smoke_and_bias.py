import sys
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.main import app

client = TestClient(app)

def load_dataset():
    """Load the telco customer churn dataset"""
    dataset_path = Path(__file__).parent.parent / "data" / "raw" / "telco-customer-churn.csv"
    return pd.read_csv(dataset_path)

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
    
    response = client.post("/predict", json=[customer_data])
    
    assert response.status_code == 200
    data = response.json()[0]
    assert "customer_id" in data
    assert "churn" in data
    assert "probs" in data
    assert "threshold_used" in data

def test_predict_invalid_data():
    """Verify predict endpoint handles invalid data properly"""
    invalid_data = {"wrong_field": "test"}
    
    response = client.post("/predict", json=invalid_data)
    
    assert response.status_code == 422  # validation error

def test_api_connectivity():
    """Quick connectivity smoke test"""
    response = client.get("/health")
    assert response.status_code == 200

def test_model_bias():
    """
    Test for potential bias in model predictions across demographic groups.
    Checks for significant disparities in churn predictions based on gender, 
    senior citizen status, and payment method.
    """
    # Load dataset
    df = load_dataset()
    
    # Define bias-sensitive features to test
    bias_features = {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'PaymentMethod': [
            'Electronic check',
            'Mailed check', 
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ]
    }
    
    # Store predictions by demographic group
    bias_metrics = defaultdict(list)
    
    # Sample size configuration (to keep test runtime reasonable)
    max_samples_per_group = 50
    
    # Test each bias-sensitive feature
    for feature, categories in bias_features.items():
        for category in categories:
            # Filter dataset for this category
            group_data = df[df[feature] == category].copy()
            
            if len(group_data) == 0:
                continue
            
            # Sample to keep test runtime manageable
            if len(group_data) > max_samples_per_group:
                group_data = group_data.sample(n=max_samples_per_group, random_state=42)
            
            # Prepare batch of customers for this demographic group
            customer_data = []
            for _, row in group_data.iterrows():
                customer_data.append({
                    "customerID": str(row['customerID']),
                    "gender": str(row['gender']),
                    "SeniorCitizen": int(row['SeniorCitizen']),
                    "Partner": str(row['Partner']),
                    "Dependents": str(row['Dependents']),
                    "tenure": int(row['tenure']),
                    "PhoneService": str(row['PhoneService']),
                    "MultipleLines": str(row['MultipleLines']),
                    "InternetService": str(row['InternetService']),
                    "OnlineSecurity": str(row['OnlineSecurity']),
                    "OnlineBackup": str(row['OnlineBackup']),
                    "DeviceProtection": str(row['DeviceProtection']),
                    "TechSupport": str(row['TechSupport']),
                    "StreamingTV": str(row['StreamingTV']),
                    "StreamingMovies": str(row['StreamingMovies']),
                    "Contract": str(row['Contract']),
                    "PaperlessBilling": str(row['PaperlessBilling']),
                    "PaymentMethod": str(row['PaymentMethod']),
                    "MonthlyCharges": float(row['MonthlyCharges']),
                    "TotalCharges": float(row['TotalCharges'])
                })
            
            # Send batch prediction for this demographic group
            try:
                response = client.post("/predict", json=customer_data)
                if response.status_code == 200:
                    predictions = response.json()
                    # Extract churn probabilities for this group
                    for prediction in predictions:
                        churn_prob = prediction['probs']['yes']
                        bias_metrics[f"{feature}_{category}"].append(churn_prob)
            except Exception as e:
                print(f"Error processing {feature}_{category}: {str(e)}")
                continue
    
    # Calculate bias metrics
    feature_disparities = {}
    bias_threshold = 0.15  # Maximum acceptable disparity
    
    for feature, categories in bias_features.items():
        category_churn_rates = {}
        
        for category in categories:
            key = f"{feature}_{category}"
            if key in bias_metrics and bias_metrics[key]:
                category_churn_rates[category] = {
                    'mean': np.mean(bias_metrics[key]),
                    'std': np.std(bias_metrics[key]),
                    'count': len(bias_metrics[key])
                }
        
        if len(category_churn_rates) >= 2:
            # Calculate maximum disparity between groups
            rates = [metrics['mean'] for metrics in category_churn_rates.values()]
            max_disparity = max(rates) - min(rates)
            feature_disparities[feature] = {
                'category_metrics': category_churn_rates,
                'max_disparity': max_disparity
            }
    
    # Assert bias check results
    print("\n=== Bias Analysis Results ===")
    bias_found = False
    
    for feature, metrics in feature_disparities.items():
        print(f"\n{feature}:")
        for category, category_metrics in metrics['category_metrics'].items():
            print(f"  {category}: {category_metrics['mean']:.4f} ± {category_metrics['std']:.4f} "
                  f"(n={category_metrics['count']})")
        print(f"  Max disparity: {metrics['max_disparity']:.4f}")
        
        if metrics['max_disparity'] > bias_threshold:
            bias_found = True
            print(f"  ⚠️  WARNING: Disparity exceeds threshold of {bias_threshold}")
    
    # Assert that no significant bias is detected
    # Only fail if disparities are found in critical demographic features
    critical_features = ['gender', 'SeniorCitizen']
    for feature in critical_features:
        if feature in feature_disparities:
            assert feature_disparities[feature]['max_disparity'] < bias_threshold, \
                f"{feature} shows significant bias with disparity {feature_disparities[feature]['max_disparity']:.4f} > {bias_threshold}"
    
    # Check if at least 2 categories per feature were tested
    for feature, categories in bias_features.items():
        categories_tested = sum(1 for c in categories if f"{feature}_{c}" in bias_metrics and bias_metrics[f"{feature}_{c}"])
        assert categories_tested >= 2, f"Insufficient data to test bias for {feature} (need at least 2 categories)"
    
    # Optional: Provide summary statistics
    if bias_found:
        print("\n⚠️  Bias detected in some features. Review the detailed metrics above.")
    else:
        print("\n✅ No significant bias detected across demographic groups.")