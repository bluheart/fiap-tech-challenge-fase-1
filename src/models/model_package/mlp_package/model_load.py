from .flexible_model import FlexibleMLP
from .data_pipeline import CustomPreprocessor
import pandas as pd
import torch
import joblib
import sys


if '__main__' in sys.modules:
    sys.modules['__main__'].CustomPreprocessor = CustomPreprocessor
else:
    # For uvicorn/multiprocessing context
    import __main__
    __main__.CustomPreprocessor = CustomPreprocessor

# Also make it available globally for good measure
globals()['CustomPreprocessor'] = CustomPreprocessor

input_size = 23
output_size = 2

class LoadModel():
    def __init__(self, model_weights_path, transform_pipeline_path):
        self.model = FlexibleMLP(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            output_size=output_size,
            activation='relu',
            dropout=0.3,
            batch_norm=True
        )
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval() # Set to evaluation mode for inference
        self.pipeline = joblib.load(transform_pipeline_path)

    
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