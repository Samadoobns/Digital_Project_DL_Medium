import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
from datetime import datetime
import shutil
import torch.nn.functional as F
# Create results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_DL_results")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"{results_dir}_{timestamp}"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Function to save logs
def log_message(message, log_file=os.path.join(results_dir, "training_log.txt")):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")

log_message("Starting model training process")

# Data loading (keeping your original structure)
Data_path = 'C:/Users/samad/OneDrive/Bureau/visulisation_project'
list_csv_files = []
for n_dir, _, n_files in os.walk(Data_path):
   list_csv_files = [file for file in n_files if file.endswith('.csv')]
list_csv_files = sorted(list_csv_files)
liste_key_names = [os.path.splitext(name)[0][:] for name in list_csv_files]
dict_data = {}
for k, n_file in zip(liste_key_names, list_csv_files):
    log_message(f"Loading dataset: {k} --- {n_file}")
    dict_data[k] = pd.read_csv(Data_path + '/' + n_file, sep=';')

# Data preparation
X_train_df = dict_data['Dataset_numerique_20000_petites_machines']
X_test_df = dict_data['Dataset_numerique_10000_petites_machines']
y_train = X_train_df.pop('Cmoy')
y_test = X_test_df.pop('Cmoy')

log_message(f"Training data shape: {X_train_df.shape}, Test data shape: {X_test_df.shape}")

def create_features(df):
    """Create additional features from existing ones with better performance"""
    log_message("Creating additional features...")
    original_shape = df.shape
    
    # Create a copy to prevent fragmentation
    df = df.copy()
    
    # Create dictionaries to collect all new features
    new_features = {}
    
    with tqdm(total=2) as pbar:
        # Create interaction features between key columns
        for col1 in df.columns[:5]:  # Example: using first 5 columns
            for col2 in df.columns[:5]:
                if col1 != col2:
                    new_features[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-10)
                    new_features[f'{col1}_{col2}_prod'] = df[col1] * df[col2]
        pbar.update(1)
        
        # Add polynomial features for important columns
        for col in df.columns[:3]:  # Example: using first 3 columns
            new_features[f'{col}_squared'] = df[col] ** 2
            new_features[f'{col}_cubed'] = df[col] ** 3
        pbar.update(1)
    
    # Create a DataFrame from all new features and concatenate with original
    new_features_df = pd.DataFrame(new_features)
    result_df = pd.concat([df, new_features_df], axis=1)
    
    log_message(f"Feature engineering complete. Original shape: {original_shape}, New shape: {result_df.shape}")
    return result_df
# Apply feature engineering
X_train_df = create_features(X_train_df)
X_test_df = create_features(X_test_df)

# Ensure test data has all columns from train data
missing_cols = set(X_train_df.columns) - set(X_test_df.columns)
if missing_cols:
    log_message(f"Adding {len(missing_cols)} missing columns to test data")
    for col in missing_cols:
        X_test_df[col] = 0

# Ensure columns order match
X_test_df = X_test_df[X_train_df.columns]

# Save column names for later reference
column_names = X_train_df.columns.tolist()
with open(os.path.join(results_dir, "feature_names.json"), "w") as f:
    json.dump(column_names, f)

# Data normalization - Using more robust scaling
log_message("Applying data transformations...")
power_transformer = PowerTransformer(method='yeo-johnson')
X_train = power_transformer.fit_transform(X_train_df)
X_test = power_transformer.transform(X_test_df)

# Target transformation - often helps regression tasks
y_transformer = PowerTransformer(method='yeo-johnson')
y_train_transformed = y_transformer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_original = y_test.copy()  # Keep original for final evaluation

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_transformed.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create validation set from training data
log_message("Creating validation split...")
X_train_tensor_split, X_val_tensor, y_train_tensor_split, y_val_tensor = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.15, random_state=42)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor_split, y_train_tensor_split)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Improved model architecture with dropout and batch normalization
class AdvancedRegressionMLP(nn.Module):
    def __init__(self, input_size):
        super(AdvancedRegressionMLP, self).__init__()
        
        # Wider and deeper network with residual connections
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # First block
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.35)
        
        # Second block
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.35)
        
        # Third block with residual connection
        self.fc3a = nn.Linear(1024, 1024)
        self.bn3a = nn.BatchNorm1d(1024)
        self.fc3b = nn.Linear(1024, 1024)
        self.bn3b = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.35)
        
        # Fourth block
        self.fc4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.3)
        
        # Fifth block with residual connection
        self.fc5a = nn.Linear(512, 512)
        self.bn5a = nn.BatchNorm1d(512)
        self.fc5b = nn.Linear(512, 512)
        self.bn5b = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.3)
        
        # Output layers
        self.fc6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc8 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.fc9 = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # Input normalization
        x = self.bn_input(x)
        
        # First block
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Third block with residual connection
        residual = x
        x = F.relu(self.bn3a(self.fc3a(x)))
        x = self.bn3b(self.fc3b(x))
        x = F.relu(x + residual)  # Residual connection
        x = self.dropout3(x)
        
        # Fourth block
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        # Fifth block with residual connection
        residual = x
        x = F.relu(self.bn5a(self.fc5a(x)))
        x = self.bn5b(self.fc5b(x))
        x = F.relu(x + residual)  # Residual connection
        x = self.dropout5(x)
        
        # Output layers
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.bn8(self.fc8(x)))
        x = self.fc9(x)
        
        return x
# Enhanced training function with learning rate scheduler, early stopping and progress bars
def train_enhanced_model(model, criterion, optimizer, train_loader, val_loader, 
                        scheduler=None, patience=10, epochs=200):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Dictionary to store training metrics
    training_history = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "epochs_completed": 0,
        "early_stopped": False,
        "best_epoch": 0
    }
    
    log_message(f"Starting training for {epochs} epochs...")
    # Main epoch progress bar
    epoch_pbar = tqdm(total=epochs, desc="Training Progress", position=0)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Batch progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                         leave=False, position=1, total=len(train_loader))
        
        for inputs, targets in train_pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Update batch progress bar with current loss
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})
        
        train_pbar.close()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation", 
                        leave=False, position=1, total=len(val_loader))
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({"val_loss": f"{loss.item():.6f}"})
        
        val_pbar.close()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        training_history["train_losses"].append(float(avg_train_loss))
        training_history["val_losses"].append(float(avg_val_loss))
        training_history["learning_rates"].append(float(current_lr))
        training_history["epochs_completed"] = epoch + 1
        
        # Update epoch progress bar with losses
        epoch_postfix = {
            "train_loss": f"{avg_train_loss:.6f}",
            "val_loss": f"{avg_val_loss:.6f}",
            "lr": f"{current_lr:.6f}"
        }
        
        epoch_pbar.set_postfix(epoch_postfix)
        epoch_pbar.update(1)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            training_history["best_epoch"] = epoch + 1
            epoch_pbar.set_postfix({**epoch_postfix, "saved": "✓"})
            
            # Save checkpoint for best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }, os.path.join(results_dir, "best_model_checkpoint.pt"))
            
           # log_message(f"Epoch {epoch+1}: New best model saved (val_loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_message(f"Early stopping at epoch {epoch+1}")
                training_history["early_stopped"] = True
                model.load_state_dict(best_model_state)
                break
    
    epoch_pbar.close()
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save training history
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f)
    
    return model, train_losses, val_losses

# Model initialization
input_size = X_train.shape[1]
log_message(f"Creating model with input size: {input_size}")
model = AdvancedRegressionMLP(input_size)

# Save model architecture description
model_summary = str(model)
with open(os.path.join(results_dir, "model_architecture.txt"), "w") as f:
    f.write(model_summary)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

# Train the model
model, train_losses, val_losses = train_enhanced_model(
    model, criterion, optimizer, train_loader, val_loader, scheduler, patience=15, epochs=200)

# Plot training history
log_message("Creating training history plot...")
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_history.png"))
plt.close()

# Evaluate on test data
log_message("Evaluating model on test data...")
model.eval()
with torch.no_grad():
    y_pred_transformed = model(X_test_tensor).numpy().flatten()

# Inverse transform predictions back to original scale
y_pred = y_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

# Calculate metrics
r2 = r2_score(y_test_original, y_pred)
mse = np.mean((y_test_original - y_pred) ** 2)
mae = np.mean(np.abs(y_test_original - y_pred))

results_message = f"\nFinal Model Performance:\nR² Score: {r2:.4f}\nMSE: {mse:.4f}\nMAE: {mae:.4f}"
log_message("="*50)
log_message(results_message)
log_message("="*50)

# Save metrics to JSON
metrics = {
    "r2_score": float(r2),
    "mse": float(mse),
    "mae": float(mae),
    "timestamp": timestamp
}
with open(os.path.join(results_dir, "model_metrics.json"), "w") as f:
    json.dump(metrics, f)

# Plot predictions vs actual
log_message("Creating prediction plot...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Predicted vs Actual Values (R² = {r2:.4f})')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "prediction_plot.png"))
plt.close()

# Create residual plot
plt.figure(figsize=(10, 6))
residuals = y_test_original - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "residual_plot.png"))
plt.close()

# Save model prediction results
results_df = pd.DataFrame({
    'actual': y_test_original,
    'predicted': y_pred,
    'residual': residuals
})
results_df.to_csv(os.path.join(results_dir, "prediction_results.csv"), index=False)

# Save the full model
log_message("Saving complete model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'power_transformer': power_transformer,
    'y_transformer': y_transformer,
    'input_size': input_size,
    'r2_score': r2,
    'feature_names': column_names
}, os.path.join(results_dir, "complete_model.pt"))

# Save a simple inference script
inference_script = """
import torch
import pickle
import pandas as pd
import numpy as np
import os

def load_model(model_path):
    # Load the model
    checkpoint = torch.load(model_path)
    
    # Get model input size
    input_size = checkpoint['input_size']
    
    # Create model architecture
    class RegressionMLP(torch.nn.Module):
        def __init__(self, input_size):
            super(RegressionMLP, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_size, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(512, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(256, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(128, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(64, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                
                torch.nn.Linear(32, 1)
            )
    
        def forward(self, x):
            return self.model(x)
    
    # Initialize model
    model = RegressionMLP(input_size)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, checkpoint['power_transformer'], checkpoint['y_transformer'], checkpoint['feature_names']

def predict(model, power_transformer, y_transformer, feature_names, input_data):
    '''
    Make predictions with the trained model
    
    Parameters:
    - model: Trained PyTorch model
    - power_transformer: Fitted feature transformer
    - y_transformer: Fitted target transformer
    - feature_names: List of feature names in the correct order
    - input_data: DataFrame or array of input features
    
    Returns:
    - Predicted values
    '''
    # Ensure input data has all required features
    if isinstance(input_data, pd.DataFrame):
        # Create missing columns if needed
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[feature_names]
    
    # Transform features
    X = power_transformer.transform(input_data)
    
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        y_pred_transformed = model(X_tensor).numpy().flatten()
    
    # Inverse transform predictions back to original scale
    y_pred = y_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
    
    return y_pred

if __name__ == "__main__":
    # Path to the saved model
    model_path = "complete_model.pt"
    
    # Load the model and transformers
    model, power_transformer, y_transformer, feature_names = load_model(model_path)
    
    # Example: Load test data (replace with your data loading code)
    test_data = pd.read_csv("your_test_data.csv", sep=";")
    
    # If there's a target column in the test data that needs to be removed
    if "Cmoy" in test_data.columns:
        y_test = test_data.pop("Cmoy")
    
    # Make predictions
    predictions = predict(model, power_transformer, y_transformer, feature_names, test_data)
    
    # Print or save predictions
    print("Predictions:", predictions[:10])  # Print first 10 predictions
    
    # Save predictions to a CSV file
    pd.DataFrame(predictions, columns=["predicted_value"]).to_csv("predictions.csv", index=False)
"""

with open(os.path.join(results_dir, "inference.py"), "w") as f:
    f.write(inference_script)

# Create a simple README
readme_content = f"""# Optimized Regression Model

## Model Performance
- R² Score: {r2:.4f}
- MSE: {mse:.4f}
- MAE: {mae:.4f}

## Directory Contents
- `complete_model.pt`: The trained model with transformers and metadata
- `model_metrics.json`: Performance metrics in JSON format
- `training_history.json`: Detailed training history
- `training_history.png`: Plot of training and validation loss
- `prediction_plot.png`: Plot of predicted vs actual values
- `residual_plot.png`: Plot of residuals
- `prediction_results.csv`: CSV file with actual values, predictions, and residuals
- `model_architecture.txt`: Text description of the model architecture
- `feature_names.json`: List of feature names used by the model
- `training_log.txt`: Detailed training log
- `inference.py`: Example script for making predictions with the model

## Usage
1. Load the model using `torch.load('complete_model.pt')`
2. See `inference.py` for an example of how to use the model for predictions

## Training Details
- Input features: {input_size}
- Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Best model saved at epoch: {val_losses.index(min(val_losses)) + 1}
"""

with open(os.path.join(results_dir, "README.md"), "w") as f:
    f.write(readme_content)

log_message(f"\nAll results saved to: {results_dir}")
log_message("Model training and evaluation complete!")