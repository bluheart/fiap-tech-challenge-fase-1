import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

class CustomerChurnSchema(pa.DataFrameModel):
    """Schema for customer churn dataset validation"""
    
    customerID: Column(pa.String, nullable=False, unique=True)
    gender: Column(pa.String, nullable=False, checks=[
        Check.isin(['Male', 'Female'])
    ])
    SeniorCitizen: Column(pa.Int, nullable=False, checks=[
        Check.isin([0, 1]),
        Check(lambda x: x >= 0, element_wise=True),
        Check(lambda x: x <= 1, element_wise=True)
    ])
    Partner: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No'])
    ])
    Dependents: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No'])
    ])
    tenure: Column(pa.Int, nullable=False, checks=[
        Check.ge(0),
        Check.le(72)
    ])
    PhoneService: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No'])
    ])
    MultipleLines: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No', 'No phone service'])
    ])
    InternetService: Column(pa.String, nullable=False, checks=[
        Check.isin(['DSL', 'Fiber optic', 'No'])
    ])
    OnlineSecurity: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No', 'No internet service'])
    ])
    OnlineBackup: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No', 'No internet service'])
    ])
    DeviceProtection: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No', 'No internet service'])
    ])
    TechSupport: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No', 'No internet service'])
    ])
    StreamingTV: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No', 'No internet service'])
    ])
    StreamingMovies: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No', 'No internet service'])
    ])
    Contract: Column(pa.String, nullable=False, checks=[
        Check.isin(['Month-to-month', 'One year', 'Two year'])
    ])
    PaperlessBilling: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No'])
    ])
    PaymentMethod: Column(pa.String, nullable=False, checks=[
        Check.isin(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
                   'Credit card (automatic)'])
    ])
    MonthlyCharges: Column(pa.Float, nullable=False, checks=[
        Check.ge(0),
        Check.le(150)
    ])
    TotalCharges: Column(pa.String, nullable=False, checks=[
        Check(lambda s: s.replace('.', '').replace('-', '').isdigit() or s == '', 
              element_wise=True, 
              error="TotalCharges must be a numeric string or empty")
    ])
    Churn: Column(pa.String, nullable=False, checks=[
        Check.isin(['Yes', 'No'])
    ])
    
    class Config:
        """Pandera configuration"""
        strict = True
        coerce = True
        
    @pa.dataframe_check
    def total_charges_consistency(cls, df: pd.DataFrame) -> bool:
        """Custom check: TotalCharges should be consistent with tenure and MonthlyCharges"""
        df_valid = df.copy()
        df_valid['TotalCharges_numeric'] = pd.to_numeric(df_valid['TotalCharges'], errors='coerce').fillna(0)
        
        expected_total = df_valid['tenure'] * df_valid['MonthlyCharges']
        
        is_consistent = (abs(df_valid['TotalCharges_numeric'] - expected_total) / 
                        (expected_total + 1e-10)) <= 0.05
        
        return is_consistent.all()