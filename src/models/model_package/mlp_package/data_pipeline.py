import pandas as pd
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import LabelEncoder

class BoolToIntTransformer(BaseEstimator, TransformerMixin):
    """Convert boolean columns to int (0/1)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        bool_cols = X_copy.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            X_copy[col] = X_copy[col].astype(int)
        return X_copy


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessing for the telecom dataset"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.mapping = {"Yes": True, "No": False, "Male": True, "Female": False}
        self.no_yes_cols = None
        self.columns_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.categorical_cols = ["InternetService", "Contract", "PaymentMethod"]

    def fit(self, X, y=None):
        # Store column names for later
        self.feature_names_in_ = X.columns if hasattr(X, "columns") else None

        # Fit scaler on numeric columns (will be applied after preprocessing)
        # Note: We need to simulate the full preprocessing to fit scaler properly
        X_temp = self._preprocess(X, fit_mode=True)
        self.scaler.fit(X_temp[self.columns_to_scale])
        return self

    def _preprocess(self, X, fit_mode=False):
        """Apply all preprocessing steps"""
        X_copy = X.copy()

        # Convert TotalCharges to numeric
        if "TotalCharges" in X_copy.columns:
            X_copy["TotalCharges"] = pd.to_numeric(
                X_copy["TotalCharges"], errors="coerce"
            )
            X_copy["TotalCharges"] = X_copy["TotalCharges"].fillna(
                X_copy["TotalCharges"].mean()
            )

        # Transform MultipleLines
        if "MultipleLines" in X_copy.columns:
            X_copy["MultipleLines"] = X_copy["MultipleLines"].apply(
                lambda value: "No" if value == "No phone service" else value
            )

        # Transform internet service related columns
        columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        for col in columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].apply(
                    lambda value: "No" if value == "No internet service" else value
                )

        # Apply mapping for Yes/No and Male/Female columns
        no_yes_cols = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "PaperlessBilling",
            "Churn",
        ] + columns
        no_yes_cols = [col for col in no_yes_cols if col in X_copy.columns]

        for col in no_yes_cols:
            X_copy[col] = X_copy[col].map(lambda value: self.mapping.get(value, value))

        # Convert SeniorCitizen to bool
        if "SeniorCitizen" in X_copy.columns:
            X_copy["SeniorCitizen"] = X_copy["SeniorCitizen"].apply(
                lambda value: bool(value)
            )

        # One-hot encoding for categorical columns
        categorical_cols = [
            col for col in self.categorical_cols if col in X_copy.columns
        ]
        if categorical_cols:
            X_copy = pd.get_dummies(X_copy, columns=categorical_cols, drop_first=True)

        return X_copy

    def transform(self, X):
        # Apply preprocessing
        X_processed = self._preprocess(X)

        # Scale numeric columns
        X_processed[self.columns_to_scale] = self.scaler.transform(
            X_processed[self.columns_to_scale]
        )

        # Convert boolean to int
        bool_cols = X_processed.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            X_processed[col] = X_processed[col].astype(int)

        return X_processed
    
    @staticmethod
    def to_tensor(X):
        """
        Convert input data to a PyTorch tensor.
        
        This method converts pandas DataFrame objects to numpy arrays before
        creating a PyTorch tensor. Other input types are directly converted
        to tensors.
        
        Args:
            X (pd.DataFrame or array-like): Input data to convert. Can be a pandas
                DataFrame, numpy array, or any other array-like structure that
                can be converted to a PyTorch tensor.
        
        Returns:
            torch.Tensor: Input data converted to a float32 PyTorch tensor.
        """
        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array with float dtype
            X = X.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        elif isinstance(X, np.ndarray):
            # If it's a numpy array but with object dtype, try to convert to float
            if X.dtype == np.object_:
                X = X.astype(np.float32)
        return torch.tensor(X, dtype=torch.float32)


# Create the complete pipeline
def create_preprocessing_pipeline():
    """
    Creates a complete preprocessing pipeline for transforming input data.
    
    The pipeline consists of two main steps:
    1. CustomPreprocessor: Applies custom preprocessing operations to the input data
    2. FunctionTransformer: Converts the preprocessed data into tensor format
    
    Returns:
        sklearn.pipeline.Pipeline: A pipeline object that sequentially applies
        the custom preprocessor and tensor converter transformations.
    """
    pipeline = Pipeline(
        [
            ("preprocessor", CustomPreprocessor()),
            ("tensor_converter", FunctionTransformer(CustomPreprocessor.to_tensor)),
        ]
    )
    return pipeline


def preprocess_label(label_train, label_test):
    """
    Encodes categorical labels into numerical format using LabelEncoder.
    
    This function fits a LabelEncoder on the training labels and then transforms
    both training and testing labels into encoded integer values.
    
    Args:
        label_train (array-like of shape (n_samples,)): Training labels to be encoded.
        label_test (array-like of shape (n_samples,)): Testing labels to be encoded.
    
    Returns:
        tuple: A tuple containing:
            - y_train (numpy.ndarray): Encoded training labels
            - y_test (numpy.ndarray): Encoded testing labels
    
    Raises:
        ValueError: If label_test contains labels not present in label_train.
    """
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(label_train)
    y_test = encoder.transform(label_test)
    return y_train, y_test