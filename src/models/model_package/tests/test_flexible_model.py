import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from model_package import (
    FlexibleMLP, EarlyStopping
)

class TestFlexibleMLP:
    """Test suite for FlexibleMLP class"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return {
            'input_size': 10,
            'hidden_sizes': [5, 3],
            'output_size': 2,
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': False
        }
    
    @pytest.fixture
    def sample_batch(self):
        """Sample batch data for testing"""
        return torch.randn(32, 10)
    
    def test_initialization_basic(self, basic_config):
        """Test basic model initialization"""
        model = FlexibleMLP(**basic_config)
        
        assert isinstance(model, FlexibleMLP)
        assert isinstance(model.network, torch.nn.Sequential)
        
        # Check number of layers (2 hidden + output)
        # Each hidden layer: Linear + Activation (+ optional BN/Dropout)
        expected_layers = 2 * 2 + 1  # 2 hidden: each has Linear+Activation, plus output Linear
        assert len(model.network) == expected_layers
    
    def test_initialization_with_batch_norm(self, basic_config):
        """Test initialization with batch normalization"""
        basic_config['batch_norm'] = True
        model = FlexibleMLP(**basic_config)
        
        # Each hidden layer: Linear + BatchNorm + Activation
        expected_layers = 2 * 3 + 1  # +1 for output layer
        assert len(model.network) == expected_layers
        assert isinstance(model.network[1], torch.nn.BatchNorm1d)
    
    def test_initialization_with_dropout(self, basic_config):
        """Test initialization with dropout"""
        basic_config['dropout'] = 0.5
        model = FlexibleMLP(**basic_config)
        
        # Each hidden layer: Linear + Activation + Dropout
        dropout_layers = [layer for layer in model.network if isinstance(layer, torch.nn.Dropout)]
        assert len(dropout_layers) == len(basic_config['hidden_sizes'])
        assert dropout_layers[0].p == 0.5
    
    def test_all_activations(self, basic_config):
        """Test all activation function options"""
        activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        
        for activation in activations:
            basic_config['activation'] = activation
            model = FlexibleMLP(**basic_config)
            
            # Check activation layer exists
            activation_layers = [layer for layer in model.network 
                               if isinstance(layer, (torch.nn.ReLU, torch.nn.Tanh, 
                                                     torch.nn.Sigmoid, torch.nn.LeakyReLU, 
                                                     torch.nn.ELU))]
            assert len(activation_layers) == len(basic_config['hidden_sizes'])
    
    def test_forward_pass_shape(self, basic_config, sample_batch):
        """Test forward pass output shape"""
        model = FlexibleMLP(**basic_config)
        output = model(sample_batch)
        
        expected_shape = (32, basic_config['output_size'])
        assert output.shape == expected_shape
        assert output.dtype == torch.float32
    
    def test_forward_pass_single_sample(self, basic_config):
        """Test forward pass with single sample"""
        model = FlexibleMLP(**basic_config)
        single_sample = torch.randn(basic_config['input_size'])
        output = model(single_sample)
        
        expected_shape = (basic_config['output_size'],)
        assert output.shape == expected_shape
    
    def test_predict_with_tensor_input(self, basic_config):
        """Test predict method with tensor input"""
        model = FlexibleMLP(**basic_config)
        x = torch.randn(16, basic_config['input_size'])

        predictions = model.predict(x, return_probs=False)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (16,)

        probabilities = model.predict(x, return_probs=True)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (16, basic_config['output_size'])  # Fixed: should be (16, 2)
        # Check that probabilities sum to 1 for each sample
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        # Check that probabilities are between 0 and 1
        assert np.all((probabilities >= 0) & (probabilities <= 1))
    
    def test_predict_with_numpy_input(self, basic_config):
        """Test predict method with numpy array input"""
        model = FlexibleMLP(**basic_config)
        x = np.random.randn(16, basic_config['input_size'])
        
        predictions = model.predict(x, return_probs=False)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (16,)
    
    def test_predict_with_list_input(self, basic_config):
        """Test predict method with list input"""
        model = FlexibleMLP(**basic_config)
        x = [[1.0] * basic_config['input_size']] * 16
        
        predictions = model.predict(x, return_probs=False)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (16,)
    
    def test_predict_single_sample(self, basic_config):
        """Test predict with single sample"""
        model = FlexibleMLP(**basic_config)
        x = torch.randn(basic_config['input_size'])
        
        prediction = model.predict_single(x, return_probs=False)
        assert isinstance(prediction, (int, np.integer))
        
        probabilities = model.predict_single(x, return_probs=True)
        assert probabilities.shape == (basic_config['output_size'],)
    
    def test_predict_with_confidence(self, basic_config):
        """Test predict_with_confidence method"""
        model = FlexibleMLP(**basic_config)
        x = torch.randn(basic_config['input_size'])
        
        predicted_class, confidence, probs = model.predict_with_confidence(x)
        
        assert isinstance(predicted_class, (int, np.integer))
        assert 0 <= confidence <= 1
        assert probs.shape == (basic_config['output_size'],)
        assert probs[predicted_class] == confidence
    
    def test_predict_invalid_input(self, basic_config):
        """Test predict with invalid input type"""
        model = FlexibleMLP(**basic_config)
        
        with pytest.raises(TypeError):
            model.predict("invalid_input")
    
    def test_model_eval_mode_during_prediction(self, basic_config):
        """Test that model is in eval mode during prediction"""
        model = FlexibleMLP(**basic_config)
        x = torch.randn(16, basic_config['input_size'])
        
        assert model.training is True
        model.predict(x)
        assert model.training is True  # Should revert to original mode
    
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    def test_predict_device_handling(self, basic_config, device):
        """Test device handling in predict method"""
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = FlexibleMLP(**basic_config)
        x = torch.randn(16, basic_config['input_size'])
        
        predictions = model.predict(x, device=device)
        assert isinstance(predictions, np.ndarray)
    
    def test_gradient_flow(self, basic_config, sample_batch):
        """Test that gradients flow correctly"""
        model = FlexibleMLP(**basic_config)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create dummy targets
        targets = torch.randint(0, basic_config['output_size'], (32,))
        
        output = model(sample_batch)
        loss = criterion(output, targets)
        loss.backward()
        
        # Check that gradients exist for parameters
        for param in model.parameters():
            assert param.grad is not None
    
    def test_empty_hidden_layers(self):
        """Test model with no hidden layers"""
        model = FlexibleMLP(input_size=10, hidden_sizes=[], output_size=2)
        
        # Should only have output layer
        assert len(model.network) == 1
        assert isinstance(model.network[0], torch.nn.Linear)
        assert model.network[0].in_features == 10
        assert model.network[0].out_features == 2


class TestEarlyStopping:
    """Test suite for EarlyStopping class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.state_dict.return_value = {'weights': torch.randn(10, 5)}
        return model
    
    @pytest.fixture
    def temp_path(self):
        """Create temporary file path for model saving"""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            yield tmp.name
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    def test_initialization(self, temp_path):
        """Test early stopping initialization"""
        early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.01, path=temp_path)
        
        assert early_stopping.patience == 5
        assert early_stopping.verbose
        assert early_stopping.delta == 0.01
        assert early_stopping.path == temp_path
        assert early_stopping.counter == 0
        assert early_stopping.best_score is None
        assert not early_stopping.early_stop
        assert early_stopping.best_accuracy == 0
    
    def test_first_call_saves_model(self, mock_model, temp_path):
        """Test that first call saves the model"""
        early_stopping = EarlyStopping(patience=3, verbose=False, path=temp_path)
        
        with patch('torch.save') as mock_save:
            early_stopping(0.95, mock_model)
            
            assert early_stopping.best_score == 0.95
            assert early_stopping.counter == 0
            mock_save.assert_called_once_with(mock_model.state_dict(), temp_path)
    
    def test_improvement_resets_counter(self, mock_model, temp_path):
        """Test that improvement resets the counter"""
        early_stopping = EarlyStopping(patience=3, verbose=False, path=temp_path)
        
        with patch('torch.save') as mock_save:
            early_stopping(0.90, mock_model)
            early_stopping(0.92, mock_model)  # Improvement
            
            assert early_stopping.best_score == 0.92
            assert early_stopping.counter == 0
            # Should save again on improvement
            assert mock_save.call_count == 2
    
    def test_no_improvement_increments_counter(self, mock_model, temp_path):
        """Test that no improvement increments counter"""
        early_stopping = EarlyStopping(patience=3, verbose=False, delta=0, path=temp_path)
        
        with patch('torch.save') as mock_save:
            early_stopping(0.90, mock_model)
            early_stopping(0.89, mock_model)  # No improvement
            
            assert early_stopping.counter == 1
            # Should only save once (first call only)
            assert mock_save.call_count == 1
    
    def test_patience_triggers_early_stop(self, mock_model, temp_path):
        """Test that exceeding patience triggers early stop"""
        early_stopping = EarlyStopping(patience=2, verbose=False, delta=0, path=temp_path)
        
        with patch('torch.save'):
            early_stopping(0.90, mock_model)
            early_stopping(0.89, mock_model)
            early_stopping(0.88, mock_model)
            
            assert early_stopping.early_stop
    
    def test_delta_ignores_small_improvements(self, mock_model, temp_path):
        """Test that delta parameter ignores small improvements"""
        early_stopping = EarlyStopping(patience=2, verbose=False, delta=0.05, path=temp_path)
        
        with patch('torch.save') as mock_save:
            early_stopping(0.90, mock_model)
            early_stopping(0.92, mock_model)  # Improvement of 0.02 < delta
            
            assert early_stopping.counter == 1
            # Should not save on small improvement
            assert mock_save.call_count == 1
    
    def test_improvement_with_large_delta_saves_model(self, mock_model, temp_path):
        """Test that improvement larger than delta saves model"""
        early_stopping = EarlyStopping(patience=2, verbose=False, delta=0.05, path=temp_path)
        
        with patch('torch.save') as mock_save:
            early_stopping(0.90, mock_model)
            early_stopping(0.96, mock_model)  # Improvement of 0.06 > delta
            
            assert early_stopping.best_score == 0.96
            assert early_stopping.counter == 0
            # Should save on significant improvement
            assert mock_save.call_count == 2
    
    def test_verbose_output_on_improvement(self, mock_model, temp_path, capsys):
        """Test verbose output when improvement occurs"""
        early_stopping = EarlyStopping(patience=3, verbose=True, path=temp_path)
        
        with patch('torch.save'):
            early_stopping(0.90, mock_model)
            captured = capsys.readouterr()
            assert "improved" in captured.out
            assert "0.0000" in captured.out  # Initial best_accuracy
    
    def test_verbose_output_on_counter(self, mock_model, temp_path, capsys):
        """Test verbose output when no improvement"""
        early_stopping = EarlyStopping(patience=3, verbose=True, delta=0, path=temp_path)
        
        with patch('torch.save'):
            early_stopping(0.90, mock_model)
            early_stopping(0.89, mock_model)
            captured = capsys.readouterr()
            assert "EarlyStopping counter: 1 out of 3" in captured.out
    
    def test_best_accuracy_tracks_best_score(self, mock_model, temp_path):
        """Test that best_accuracy is updated correctly"""
        early_stopping = EarlyStopping(patience=3, verbose=False, path=temp_path)
        
        with patch('torch.save'):
            early_stopping(0.85, mock_model)
            assert early_stopping.best_accuracy == 0.85
            
            early_stopping(0.90, mock_model)
            assert early_stopping.best_accuracy == 0.90
            
            early_stopping(0.88, mock_model)  # No improvement
            assert early_stopping.best_accuracy == 0.90  # Should stay at best
    
    def test_save_checkpoint_called_on_improvement(self, mock_model, temp_path):
        """Test that save_checkpoint logic is triggered on improvement"""
        early_stopping = EarlyStopping(patience=3, verbose=False, path=temp_path)
        
        with patch('torch.save') as mock_save:
            # First call - should save
            early_stopping(0.85, mock_model)
            assert mock_save.call_count == 1
            
            # Second call - improvement, should save
            early_stopping(0.90, mock_model)
            assert mock_save.call_count == 2
            
            # Third call - no improvement, should not save
            early_stopping(0.89, mock_model)
            assert mock_save.call_count == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])