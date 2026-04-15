from .flexible_model import FlexibleMLP
from .customer_churn_dataframe_schema import CustomerChurnSchema
from .data_pipeline import CustomPreprocessor
import pandera as pa
import pandas as pd
import torch
import joblib

input_size = 23
output_size = 2
model_path = './models/model_weights_v1.pth'
pipeline_path = './models/data_processing.joblib'

class LoadModel():
    def __init__(self):
        self.model = FlexibleMLP(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            output_size=output_size,
            activation='relu',
            dropout=0.3,
            batch_norm=True
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval() # Set to evaluation mode for inference
        self.pipeline = joblib.load()

    
    def predict(self, customer: dict, threshold=0.20):
        df = pd.DataFrame(customer)
        """
        try:
            df = CustomerChurnSchema.validate(df)
        except pa.errors.SchemaError as e:
            print(f"Validation failed: {e}")
            raise
        """
        df = df.drop(columns=['customerID'])
        tensor = self.pipeline.transform(df)
        predictions = self.model.predict(tensor, return_probs=True)
        predictions = (predictions >= threshold).astype(int)
        return predictions