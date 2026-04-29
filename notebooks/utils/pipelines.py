import pandas as pd
import torch
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(filename)s | %(lineno)s | %(funcName)20s() %(message)s'
)

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
        self.one_hot_columns = [
            "InternetService_Fiber optic",
            "InternetService_No",
            "Contract_One year",
            "Contract_Two year",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check"
        ]
        
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
        
        logging.debug(f"Before One-hot {X_copy.columns}")

        for cat_col in self.categorical_cols:
            if cat_col in X_copy.columns:
                # For each possible category in our hard-coded list, create a column
                for oh_col in self.one_hot_columns:
                    if oh_col.startswith(f"{cat_col}_"):
                        # Extract the category value (everything after the underscore)
                        category_value = oh_col.split('_', 1)[1]
                        # Create column with 1 if matches, else 0
                        X_copy[oh_col] = (X_copy[cat_col] == category_value).astype(int)
                
                # Drop the original categorical column
                X_copy = X_copy.drop(columns=[cat_col])

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
        # Convert to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return torch.tensor(X, dtype=torch.float32)


# Create the complete pipeline
def create_preprocessing_pipeline():
    pipeline = Pipeline(
        [
            ("preprocessor", CustomPreprocessor()),
            ("tensor_converter", FunctionTransformer(CustomPreprocessor.to_tensor)),
        ]
    )
    return pipeline

def preprocess_label(label_train, label_test):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(label_train)
    y_test = encoder.transform(label_test)
    return y_train, y_test

# train pipeline if run
if __name__ == "__main__":
    RANDOM_SEED = 19
    dados = "data/raw/telco-customer-churn.csv"
    local_modelo = 'artifacts/models/data_processing.joblib'
    df = pd.read_csv(dados)

    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(X_train)
    logging.info(f"Pipeline de transformação de dados treinada com arquivo {dados}")
    joblib.dump(pipeline, local_modelo)
    logging.info(f"Joblib salva em {local_modelo}")