import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import your module
from mlp_package import LoadModel, FlexibleMLP

model_path = './models/model_weights_v1.pth'
pipeline_path = './models/data_processing.joblib'

class TestLoadModel:
    """Simplified tests for LoadModel class"""
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing"""
        with patch('joblib.load') as mock_joblib, \
             patch('torch.load') as mock_torch, \
             patch('mlp_package.model_load.FlexibleMLP') as mock_mlp_class:
            
            # Create mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline.transform.return_value = np.array([[1, 2, 3]])
            mock_joblib.return_value = mock_pipeline
            
            # Create mock neural network
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0])
            mock_mlp_class.return_value = mock_model
            
            # Mock torch load
            mock_torch.return_value = {'state_dict': {}}
            
            yield {
                'pipeline': mock_pipeline,
                'model': mock_model,
                'joblib': mock_joblib,
                'torch': mock_torch,
                'mlp_class': mock_mlp_class
            }
    
    @pytest.fixture
    def load_model(self, mock_models, tmp_path):
        """Create LoadModel instance"""
        # Create dummy model files
        model_file = tmp_path / 'model.pth'
        model_file.touch()
        
        # Initialize LoadModel
        return LoadModel(model_path, pipeline_path)
    
    def test_initialization(self, load_model):
        """Test that LoadModel initializes correctly"""
        assert load_model is not None
    
    def test_predict_single_customer(self, load_model, mock_models):
        """Test prediction for a single customer"""
        # Sample customer data
        customer = {
            'customerID': '7590-VHVEG',
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'Yes',
            'tenure': 24,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'Yes',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 85.5,
            'TotalCharges': '2052.0'
        }
        
        # Mock the model prediction
        mock_models['model'].predict.return_value = np.array([1])
        
        # Test prediction
        result = load_model.predict([customer])
        
        # Assertions
        assert result in [0, 1] or result in ['Yes', 'No']
        mock_models['model'].predict.assert_called_once()


class TestFlexibleMLP:
    """Simplified tests for FlexibleMLP model"""
    
    @pytest.fixture
    def mlp_model(self):
        """Create a simple MLP model for testing"""
        input_dim = 10
        hidden_dims = [20, 15]
        output_dim = 2
        
        return FlexibleMLP(input_dim, hidden_dims, output_dim)
    
    def test_model_initialization(self, mlp_model):
        """Test that model initializes correctly"""
        assert mlp_model is not None
        assert hasattr(mlp_model, 'forward')
    
    def test_forward_pass(self, mlp_model):
        """Test forward pass through the network"""
        # Create random input
        batch_size = 5
        input_dim = 10
        x = np.random.randn(batch_size, input_dim).astype(np.float32)
        
        # Convert to tensor if needed
        import torch
        x_tensor = torch.from_numpy(x)
        
        # Forward pass
        output = mlp_model.forward(x_tensor)
        
        # Assertions
        assert output.shape[0] == batch_size
        assert output.shape[1] == 2  # output_dim
    
    def test_model_prediction(self, mlp_model):
        """Test model prediction method"""
        # Create sample input
        sample = np.random.randn(1, 10).astype(np.float32)
        
        # Get prediction
        prediction = mlp_model.predict(sample)
        
        # Assertions
        assert prediction in [0, 1]