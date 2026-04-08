import torch
from collections import defaultdict
import torch.nn as nn
import numpy as np

class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, 
                 activation='relu', dropout=0.0, batch_norm=False):
        super(FlexibleMLP, self).__init__()
        
        # Activation function mapping
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization (optional)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(activations[activation])
            
            # Dropout (optional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer (no activation - use with CrossEntropyLoss)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, x, return_probs=False, device=None):
        """
        Make predictions with the trained model
        
        Args:
            x: Input data - can be:
                - torch.Tensor of shape (input_size,) or (batch_size, input_size)
                - numpy array of shape (input_size,) or (batch_size, input_size)
                - list of shape (input_size,) or (batch_size, input_size)
            return_probs (bool): If True, return class probabilities.
                                 If False, return class indices (default: False)
            device (str, optional): Device to run prediction on ('cpu' or 'cuda').
                                    If None, uses current model's device.
        
        Returns:
            If return_probs=False: 
                - For single sample: int (class index)
                - For batch: numpy array of shape (batch_size,) with class indices
            If return_probs=True:
                - For single sample: numpy array of shape (num_classes,)
                - For batch: numpy array of shape (batch_size, num_classes)
        """
        # Set model to evaluation mode
        self.eval()
        
        # Determine device
        if device is None:
            device = next(self.parameters()).device
        
        # Convert input to tensor and handle different input types
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(x)}. Expected torch.Tensor, numpy.ndarray, or list.")
        
        # Track if this is a single sample (no batch dimension)
        is_single_sample = (x.dim() == 1)
        
        # Add batch dimension if needed
        if is_single_sample:
            x = x.unsqueeze(0)
        
        # Move to device
        x = x.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self(x)
            
            if return_probs:
                # Apply softmax to get probabilities
                predictions = torch.softmax(outputs, dim=1)[:, 1].numpy()
            else:
                # Get class indices
                predictions = torch.argmax(outputs, dim=1)
        
        # Convert to numpy and handle single sample case
        if is_single_sample:
            if return_probs:
                return predictions.squeeze(0)
            else:
                return predictions.item()
        else:
            return predictions
    
    def predict_single(self, x, return_probs=False, device=None):
        """
        Convenience method for predicting a single sample
        
        Args:
            x: Input data - can be torch.Tensor, numpy array, or list of shape (input_size,)
            return_probs (bool): If True, return class probabilities (default: False)
            device (str, optional): Device to run prediction on
        
        Returns:
            If return_probs=False: int (class index)
            If return_probs=True: numpy array of shape (num_classes,)
        """
        return self.predict(x, return_probs=return_probs, device=device)
    
    def predict_with_confidence(self, x, device=None):
        """
        Predict class with confidence score for a single sample
        
        Args:
            x: Input data of shape (input_size,)
            device (str, optional): Device to run prediction on
        
        Returns:
            tuple: (predicted_class, confidence_score, all_probabilities)
        """
        probs = self.predict(x, return_probs=True, device=device)
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        return predicted_class, confidence, probs

    

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            verbose (bool): If True, prints improvement messages
            delta (float): Minimum change to qualify as improvement
            path (str): Path for saving the checkpoint
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_accuracy = 0
        
    def __call__(self, val_accuracy, model):
        score = val_accuracy
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """Saves model when validation accuracy improves"""
        if self.verbose:
            print(f'Validation accuracy improved ({self.best_accuracy:.4f} --> {self.best_score:.4f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.best_accuracy = self.best_score

# Training with DataLoader and Early Stopping
def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, 
                              epochs=100, patience=7, device='cpu'):
    """
    Train model with early stopping and comprehensive logging
    """
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # For tracking metrics
    history = defaultdict(list)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate training accuracy for monitoring
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Early stopping check
        early_stopping(val_accuracy, model)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, history
