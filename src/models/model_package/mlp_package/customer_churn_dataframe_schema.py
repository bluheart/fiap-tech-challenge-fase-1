import pandera.pandas as pa
from pandera.typing import Series

class CustomerChurnSchema(pa.DataFrameModel):
    """Schema for customer churn data"""
    
    customerID: Series[str] = pa.Field(nullable=False)
    gender: Series[str] = pa.Field(isin=['Male', 'Female'], nullable=False)
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    Partner: Series[str] = pa.Field(isin=['Yes', 'No'], nullable=False)
    Dependents: Series[str] = pa.Field(isin=['Yes', 'No'], nullable=False)
    tenure: Series[int] = pa.Field(ge=0, le=72, nullable=False)
    PhoneService: Series[str] = pa.Field(isin=['Yes', 'No'], nullable=False)
    MultipleLines: Series[str] = pa.Field(isin=['Yes', 'No', 'No phone service'], nullable=False)
    InternetService: Series[str] = pa.Field(isin=['DSL', 'Fiber optic', 'No'], nullable=False)
    OnlineSecurity: Series[str] = pa.Field(isin=['Yes', 'No', 'No internet service'], nullable=False)
    OnlineBackup: Series[str] = pa.Field(isin=['Yes', 'No', 'No internet service'], nullable=False)
    DeviceProtection: Series[str] = pa.Field(isin=['Yes', 'No', 'No internet service'], nullable=False)
    TechSupport: Series[str] = pa.Field(isin=['Yes', 'No', 'No internet service'], nullable=False)
    StreamingTV: Series[str] = pa.Field(isin=['Yes', 'No', 'No internet service'], nullable=False)
    StreamingMovies: Series[str] = pa.Field(isin=['Yes', 'No', 'No internet service'], nullable=False)
    Contract: Series[str] = pa.Field(isin=['Month-to-month', 'One year', 'Two year'], nullable=False)
    PaperlessBilling: Series[str] = pa.Field(isin=['Yes', 'No'], nullable=False)
    PaymentMethod: Series[str] = pa.Field(
        isin=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
              'Credit card (automatic)'], 
        nullable=False
    )
    MonthlyCharges: Series[float] = pa.Field(ge=0, nullable=False)
    TotalCharges: Series[float] = pa.Field(ge=0, nullable=False)
    
    class Config:
        strict = True
        coerce = True