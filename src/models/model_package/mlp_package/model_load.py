from .flexible_model import FlexibleMLP
from .data_pipeline import CustomPreprocessor
from .customer_churn_dataframe_schema import CustomerChurnSchema
from typing import List
import pandas as pd
import torch
import joblib
import sys
import pandera as pa
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(filename)s | %(lineno)s | %(funcName)20s() %(message)s'
)

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
        logging.info(f"model: {self.model} loaded")
        self.model.eval() # Set to evaluation mode for inference
        logging.info(f"model: {self.model} set to evaluation mode")
        self.pipeline = joblib.load(transform_pipeline_path)
        logging.info(f"transformation pipeline: {self.pipeline} loaded")

    
    def predict(self, customer: List[dict], threshold=0.20):
        df = pd.DataFrame(customer)
        try:
            df = CustomerChurnSchema.validate(df)
        except pa.errors.SchemaError as e:
            logging.error(f"Validation failed: {e}")
            raise
        df = df.drop(columns=['customerID'])

        logging.debug(f"Dataframe {df} is ready\n Columns: {df.columns}")

        tensor = self.pipeline.transform(df)

        logging.debug(f"Tensor {tensor}")

        predictions = self.model.predict(tensor, return_probs=True)
        return predictions.tolist()