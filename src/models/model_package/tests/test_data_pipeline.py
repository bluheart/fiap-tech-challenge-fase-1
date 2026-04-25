import pytest
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

# Import your modules
from mlp_package import (
    BoolToIntTransformer,
    CustomPreprocessor,
    create_preprocessing_pipeline,
    preprocess_label
)


class TestBoolToIntTransformer:
    """Test suite for BoolToIntTransformer"""
    
    def test_transform_with_bool_columns(self):
        """Test transformation of boolean columns to integers"""
        transformer = BoolToIntTransformer()
        df = pd.DataFrame({
            'bool_col1': [True, False, True],
            'bool_col2': [False, True, False],
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c']
        })
        
        result = transformer.transform(df)
        
        assert result['bool_col1'].dtype == int
        assert result['bool_col2'].dtype == int
        assert result['int_col'].dtype == int
        assert result['str_col'].dtype == object
        assert result['bool_col1'].tolist() == [1, 0, 1]
        assert result['bool_col2'].tolist() == [0, 1, 0]
    
    def test_transform_without_bool_columns(self):
        """Test transformation when no boolean columns exist"""
        transformer = BoolToIntTransformer()
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        result = transformer.transform(df)
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_fit_returns_self(self):
        """Test that fit returns self"""
        transformer = BoolToIntTransformer()
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        result = transformer.fit(df)
        
        assert result is transformer
    
    def test_transform_does_not_modify_original(self):
        """Test that transform doesn't modify the original DataFrame"""
        transformer = BoolToIntTransformer()
        df = pd.DataFrame({'bool_col': [True, False, True]})
        df_copy = df.copy()
        
        transformer.transform(df)
        
        pd.testing.assert_frame_equal(df, df_copy)


class TestCustomPreprocessor:
    """Test suite for CustomPreprocessor"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample telecom data for testing"""
        return pd.DataFrame({
            'customerID': ['A', 'B', 'C', 'D', 'E'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tenure': [1, 12, 24, 36, 48],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'OnlineBackup': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'DeviceProtection': ['No', 'No', 'No internet service', 'Yes', 'No'],
            'TechSupport': ['Yes', 'No', 'No internet service', 'No', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
            'StreamingMovies': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Electronic check', 'Credit card'],
            'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
            'TotalCharges': [29.85, 683.40, '1292.40', 1522.80, 3393.60],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create a CustomPreprocessor instance"""
        return CustomPreprocessor()
    
    def test_fit_stores_feature_names(self, preprocessor, sample_data):
        """Test that fit stores feature names"""
        preprocessor.fit(sample_data)
        
        assert hasattr(preprocessor, 'feature_names_in_')
        assert preprocessor.feature_names_in_ is not None
    
    def test_preprocess_converts_total_charges(self, preprocessor, sample_data):
        """Test that TotalCharges is converted to numeric"""
        result = preprocessor._preprocess(sample_data)
        
        assert result['TotalCharges'].dtype in [np.float64, np.int64]
        assert not result['TotalCharges'].isna().any()
    
    def test_preprocess_handles_missing_total_charges(self, preprocessor):
        """Test handling of missing TotalCharges values"""
        df = pd.DataFrame({
            'TotalCharges': ['100', 'invalid', '200', None, '300'],
            'tenure': [1, 2, 3, 4, 5],
            'MonthlyCharges': [10, 20, 30, 40, 50]
        })
        
        result = preprocessor._preprocess(df)
        
        # Should fill NaN with mean of valid values
        assert result['TotalCharges'].isna().sum() == 0
        assert result['TotalCharges'].dtype in [np.float64, np.int64]
    
    def test_preprocess_transforms_multiple_lines(self, preprocessor, sample_data):
        """Test transformation of MultipleLines column"""
        result = preprocessor._preprocess(sample_data)
        assert 'No phone service' not in result['MultipleLines'].values
    
    def test_preprocess_transforms_internet_service_columns(self, preprocessor, sample_data):
        """Test transformation of internet service related columns"""
        result = preprocessor._preprocess(sample_data)
        
        # 'No internet service' should become 'No'
        for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']:
            if col in result.columns:
                assert 'No internet service' not in result[col].values
    
    def test_preprocess_applies_mapping(self, preprocessor, sample_data):
        """Test that mapping is applied to Yes/No and Male/Female columns"""
        result = preprocessor._preprocess(sample_data)
        
        # Check gender mapping
        assert result['gender'].dtype == bool
        assert result['gender'].iloc[0]  # Male -> True
        
        # Check Partner mapping
        assert result['Partner'].dtype == bool
        assert result['Partner'].iloc[0]  # Yes -> True
    
    def test_preprocess_converts_senior_citizen(self, preprocessor, sample_data):
        """Test conversion of SeniorCitizen to boolean"""
        result = preprocessor._preprocess(sample_data)
        
        assert result['SeniorCitizen'].dtype == bool
        assert result['SeniorCitizen'].tolist() == [False, True, False, True, False]
    
    def test_preprocess_one_hot_encoding(self, preprocessor, sample_data):
        """Test one-hot encoding of categorical columns"""
        result = preprocessor._preprocess(sample_data)
        
        # Check that original categorical columns are gone
        for col in preprocessor.categorical_cols:
            if col in sample_data.columns:
                assert col not in result.columns
        
        # Check that dummy columns were created
        dummy_columns = [col for col in result.columns if any(
            cat in col for cat in ['InternetService', 'Contract', 'PaymentMethod']
        )]
        assert len(dummy_columns) > 0
    
    @staticmethod
    def test_to_tensor_with_dataframe():
        """Test conversion of DataFrame to tensor"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.5, 5.5, 6.5]
        })
        
        tensor = CustomPreprocessor.to_tensor(df)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (3, 2)
        assert torch.all(tensor == torch.tensor([[1.0, 4.5], [2.0, 5.5], [3.0, 6.5]]))
    
    @staticmethod
    def test_to_tensor_with_numpy_array():
        """Test conversion of numpy array to tensor"""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        
        tensor = CustomPreprocessor.to_tensor(arr)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.all(tensor == torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    
    @staticmethod
    def test_to_tensor_with_list():
        """Test conversion of list to tensor"""
        lst = [[1, 2], [3, 4]]
        
        tensor = CustomPreprocessor.to_tensor(lst)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32


class TestPipelineCreation:
    @pytest.fixture
    def sample_data(self):
        """Create sample telecom data for testing"""
        return pd.DataFrame({
            'customerID': ['A', 'B', 'C', 'D', 'E'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tenure': [1, 12, 24, 36, 48],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'OnlineBackup': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'DeviceProtection': ['No', 'No', 'No internet service', 'Yes', 'No'],
            'TechSupport': ['Yes', 'No', 'No internet service', 'No', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
            'StreamingMovies': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Electronic check', 'Credit card'],
            'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
            'TotalCharges': [29.85, 683.40, '1292.40', 1522.80, 3393.60],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
        })
    """Test suite for create_preprocessing_pipeline function"""
    
    def test_create_preprocessing_pipeline_returns_pipeline(self):
        """Test that function returns a Pipeline object"""
        pipeline = create_preprocessing_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'preprocessor'
        assert pipeline.steps[1][0] == 'tensor_converter'
    
    def test_pipeline_fit_transform(self, sample_data):
        """Test that pipeline can fit and transform data"""
        pipeline = create_preprocessing_pipeline()
        
        # Fit the pipeline
        pipeline.fit(sample_data)
        
        # Transform
        result = pipeline.transform(sample_data)
        
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape[0] == len(sample_data)
    
    def test_pipeline_fit_transform_without_fit_first_raises_error(self, sample_data):
        """Test that transform without fit raises appropriate error"""
        pipeline = create_preprocessing_pipeline()
        
        # Should raise NotFittedError
        with pytest.raises(NotFittedError):
            pipeline.transform(sample_data)


class TestLabelPreprocessing:
    """Test suite for preprocess_label function"""
    
    def test_preprocess_label_basic_functionality(self):
        """Test basic label encoding functionality"""
        y_train = ['No', 'Yes', 'No', 'Yes', 'No']
        y_test = ['Yes', 'No', 'Yes']
        
        y_train_encoded, y_test_encoded = preprocess_label(y_train, y_test)
        
        assert isinstance(y_train_encoded, np.ndarray)
        assert isinstance(y_test_encoded, np.ndarray)
        assert len(y_train_encoded) == len(y_train)
        assert len(y_test_encoded) == len(y_test)
        assert set(y_train_encoded) == {0, 1}
        assert set(y_test_encoded) == {0, 1}
    
    def test_preprocess_label_consistent_encoding(self):
        """Test that encoding is consistent between train and test"""
        y_train = ['Low', 'Medium', 'High', 'Low', 'Medium']
        y_test = ['High', 'Low', 'Medium']
        
        y_train_encoded, y_test_encoded = preprocess_label(y_train, y_test)
        
        # Check that same labels get same encoding
        assert y_train_encoded[y_train.index('Low')] == y_test_encoded[1]  # 'Low' in test
        assert y_train_encoded[y_train.index('Medium')] == y_test_encoded[2]  # 'Medium' in test
        assert y_train_encoded[y_train.index('High')] == y_test_encoded[0]  # 'High' in test
    
    def test_preprocess_label_with_unseen_label_raises_error(self):
        """Test that unseen labels in test raise ValueError"""
        y_train = ['Cat', 'Dog', 'Cat']
        y_test = ['Bird']  # 'Bird' not in training
        
        with pytest.raises(ValueError):
            preprocess_label(y_train, y_test)
    
    def test_preprocess_label_with_empty_input(self):
        """Test handling of empty inputs"""
        y_train = []
        y_test = []
        
        y_train_encoded, y_test_encoded = preprocess_label(y_train, y_test)
        
        assert len(y_train_encoded) == 0
        assert len(y_test_encoded) == 0
    
    def test_preprocess_label_with_numpy_arrays(self):
        """Test with numpy array inputs"""
        y_train = np.array(['A', 'B', 'A', 'C'])
        y_test = np.array(['B', 'C', 'A'])
        
        y_train_encoded, y_test_encoded = preprocess_label(y_train, y_test)
        
        assert isinstance(y_train_encoded, np.ndarray)
        assert isinstance(y_test_encoded, np.ndarray)
        assert len(y_train_encoded) == 4
        assert len(y_test_encoded) == 3


class TestIntegration:
    """Integration tests for the complete preprocessing workflow"""
    
    @pytest.fixture
    def complete_dataset(self):
        """Create a complete dataset for integration testing"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'customerID': [f'CUST_{i}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(20, 5000, n_samples),
            'Churn': np.random.choice(['Yes', 'No'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_complete_workflow(self, complete_dataset):
        """Test the complete preprocessing and training workflow"""
        # Split data
        X = complete_dataset.drop(columns=['customerID', 'Churn'])
        y = complete_dataset['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and fit pipeline
        pipeline = create_preprocessing_pipeline()
        pipeline.fit(X_train)
        
        # Transform data
        X_train_tensor = pipeline.transform(X_train)
        X_test_tensor = pipeline.transform(X_test)
        
        # Encode labels
        y_train_encoded, y_test_encoded = preprocess_label(y_train, y_test)
        
        # Assertions
        assert isinstance(X_train_tensor, torch.Tensor)
        assert isinstance(X_test_tensor, torch.Tensor)
        assert X_train_tensor.shape[0] == len(X_train)
        assert X_test_tensor.shape[0] == len(X_test)
        assert X_train_tensor.shape[1] == X_test_tensor.shape[1]  # Same number of features
        assert len(y_train_encoded) == len(X_train)
        assert len(y_test_encoded) == len(X_test)
    
    def test_pipeline_persistence(self, complete_dataset, tmp_path):
        """Test that pipeline can be saved and loaded"""
        # Create and fit pipeline
        pipeline = create_preprocessing_pipeline()
        pipeline.fit(complete_dataset.drop(columns=['customerID', 'Churn']))
        
        # Save pipeline
        model_path = tmp_path / "test_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        
        # Load pipeline
        loaded_pipeline = joblib.load(model_path)
        
        # Transform with both pipelines
        X = complete_dataset.drop(columns=['customerID', 'Churn'])
        original_result = pipeline.transform(X)
        loaded_result = loaded_pipeline.transform(X)
        
        # Compare results
        assert torch.allclose(original_result, loaded_result)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])