from pydantic import BaseModel, Field, validator
from typing import Optional

class CustomerChurnBase(BaseModel):
    customerID: str = Field(..., min_length=1)
    
    # Demographics
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    
    # Account info
    tenure: int = Field(..., ge=0, le=72)
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(..., pattern=r"^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$")
    MonthlyCharges: float = Field(..., ge=0, le=150)
    TotalCharges: float = Field(..., ge=0)
    
    # Services
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(..., pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No|No internet service)$")
    
    @validator('TotalCharges')
    def validate_total_charges(cls, v, values):
        if 'MonthlyCharges' in values and 'tenure' in values:
            expected = values['MonthlyCharges'] * values['tenure']
            if not (expected * 0.9 <= v <= expected * 1.1):
                raise ValueError(f'TotalCharges should be approx {expected}')
        return v