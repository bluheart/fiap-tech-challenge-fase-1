from fastapi import FastAPI
from .schemas.predict import CustomerChurnBase
from mlp_package import LoadModel

app = FastAPI()

# Your validator schema (from previous answer)



@app.post("/predict")
async def predict_churn(customer: CustomerChurnBase):
    """
    Accepts customer data and returns churn prediction
    """
    model = LoadModel()
    print(model)
    
    return {
        "customer_id": "generated_id",
        "churn_risk": "High",  # Example output
        "validated_data": customer.dict()
    }