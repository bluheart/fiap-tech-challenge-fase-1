import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


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


def to_tensor(X):
    return torch.tensor(X, dtype=torch.float32)


# Create the complete pipeline
def create_preprocessing_pipeline():
    pipeline = Pipeline(
        [
            ("preprocessor", CustomPreprocessor()),
            ("tensor_converter", FunctionTransformer(to_tensor)),
        ]
    )
    return pipeline