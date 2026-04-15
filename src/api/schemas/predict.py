from pydantic import BaseModel, Field, validator
from typing import Optional

class CustomerChurnBase(BaseModel):
    # Demographics
    gender: str = Field(..., regex="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., regex="^(Yes|No)$")
    Dependents: str = Field(..., regex="^(Yes|No)$")
    
    # Account info
    tenure: int = Field(..., ge=0, le=72)
    Contract: str = Field(..., regex="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., regex="^(Yes|No)$")
    PaymentMethod: str = Field(..., regex="^(Electronic check|Mailed check|Bank transfer|Credit card)$")
    MonthlyCharges: float = Field(..., ge=0, le=150)
    TotalCharges: float = Field(..., ge=0)
    
    # Services
    PhoneService: str = Field(..., regex="^(Yes|No)$")
    MultipleLines: Optional[str] = Field(None, regex="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., regex="^(DSL|Fiber optic|No)$")
    OnlineSecurity: Optional[str] = Field(None, regex="^(Yes|No|No internet service)$")
    OnlineBackup: Optional[str] = Field(None, regex="^(Yes|No|No internet service)$")
    DeviceProtection: Optional[str] = Field(None, regex="^(Yes|No|No internet service)$")
    TechSupport: Optional[str] = Field(None, regex="^(Yes|No|No internet service)$")
    StreamingTV: Optional[str] = Field(None, regex="^(Yes|No|No internet service)$")
    StreamingMovies: Optional[str] = Field(None, regex="^(Yes|No|No internet service)$")
    
    # Target (for prediction)
    Churn: Optional[str] = Field(None, regex="^(Yes|No)$")
    
    @validator('TotalCharges')
    def validate_total_charges(cls, v, values):
        if 'MonthlyCharges' in values and 'tenure' in values:
            expected = values['MonthlyCharges'] * values['tenure']
            if not (expected * 0.9 <= v <= expected * 1.1):
                raise ValueError(f'TotalCharges should be approx {expected}')
        return v