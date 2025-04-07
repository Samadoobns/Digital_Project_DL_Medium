import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
from datetime import datetime
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
# Data loading code remains the same
Data_path = 'C:/Users/samad/OneDrive/Bureau/visulisation_project'
list_csv_files = []
for n_dir, _, n_files in os.walk(Data_path):
   list_csv_files = [file for file in n_files if file.endswith('.csv')]
list_csv_files = sorted(list_csv_files)
liste_key_names = [os.path.splitext(name)[0][:] for name in list_csv_files]
dict_data = {}
for k, n_file in zip(liste_key_names, list_csv_files):
    print(k,'---', n_file)
    dict_data[k] = pd.read_csv(Data_path + '/' + n_file,sep=';')

X_train = dict_data['Dataset_numerique_20000_petites_machines']
X_test = dict_data['Dataset_numerique_10000_petites_machines']
y_train = X_train.pop('Cmoy')
y_test = X_test.pop('Cmoy')

# Create a validation set
log_message("Creating validation split...")
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
log_message(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Feature engineering - add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_raw)
X_val_poly = poly.transform(X_val_raw)
X_test_poly = poly.transform(X_test)
X_train_poly = pd.DataFrame(X_train_poly)
X_val_poly = pd.DataFrame(X_val_poly)
X_test_poly = pd.DataFrame(X_test_poly)

# Ensure test data has all columns from train data
missing_cols = set(X_train_poly.columns) - set(X_test_poly.columns)
if missing_cols:
    log_message(f"Adding {len(missing_cols)} missing columns to test data")
    for col in missing_cols:
        X_test_poly[col] = 0

# Ensure columns order match
X_test_df = X_test_poly[X_train_poly.columns]

# Save column names for later reference
column_names = X_train_poly.columns.tolist()
with open(os.path.join(results_dir, "feature_names.json"), "w") as f:
    json.dump(column_names, f)
    
# Data normalization - Using more robust scaling
log_message("Applying data transformations...")
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_val_scaled = scaler.transform(X_val_poly)
X_test_scaled = scaler.transform(X_test_poly)


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw.values.reshape(-1, 1), dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_raw.values.reshape(-1, 1), dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Create data loaders for batch processing
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Improved model architecture
class EnhancedRegressionMLP(nn.Module):
    def __init__(self, input_size):
        super(EnhancedRegressionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Enhanced training function with validation
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=20, patience=30):
    best_val_loss = float('inf')
    patience_counter = 0
    # Dictionary to store training metrics
    training_history = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "val_r2": [],
        "epochs_completed": 0,
        "early_stopped": False,
        "best_epoch": 0
    }
    log_message(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Add progress bar to training loop
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training', ncols=100)
        
        for X_batch, y_batch in train_loader_tqdm:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
            # Update progress bar
            train_loader_tqdm.set_postfix(loss=train_loss / len(train_loader.dataset))
        
        train_loss /= len(train_loader.dataset)
        training_history['train_losses'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        # Add progress bar to validation loop
        val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation', ncols=100)
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader_tqdm:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_preds.append(outputs.numpy())
                val_targets.append(y_batch.numpy())
                
                # Update progress bar
                val_loader_tqdm.set_postfix(val_loss=val_loss / len(val_loader.dataset))
        
        val_loss /= len(val_loader.dataset)
        training_history['val_losses'].append(val_loss)
        
        # Calculate R² score on validation set
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_r2 = r2_score(val_targets, val_preds)
        training_history['val_r2'].append(val_r2)
        
        # Print progress
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R²: {val_r2:.4f}")
        training_history["epochs_completed"] = epoch + 1
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    #model.load_state_dict(torch.load('best_model.pt'))
     # Save the full model
    log_message("Saving complete model...")


    return model, training_history

# Initialize model
input_size = X_train_scaled.shape[1]
model = EnhancedRegressionMLP(input_size)

# Loss function, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Train model
model, history = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=20)

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).detach().numpy().flatten()
r2 = r2_score(y_test, y_pred)
print(f"Score R² final sur les données test : {r2:.4f}")
# Calculate metrics
y_test_np = y_test_tensor.numpy().flatten()  # ou y_test.values.flatten()

r2 = r2_score(y_test, y_pred)
mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean(np.abs(y_test_np - y_pred))

results_message = f"\nFinal Model Performance:\nR² Score: {r2:.4f}\nMSE: {mse:.4f}\nMAE: {mae:.4f}"
log_message("="*50)
log_message(results_message)
log_message("="*50)
            # Save best model
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'r2_score': r2,
    'feature_names': column_names
    }, os.path.join(results_dir, "best_model.pt"))
# Save metrics to JSON
metrics = {
    "r2_score": float(r2),
    "mse": float(mse),
    "mae": float(mae),
    "timestamp": timestamp
}
with open(os.path.join(results_dir, "model_metrics.json"), "w") as f:
    json.dump(metrics, f)
# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_losses'], label='Train Loss')
plt.plot(history['val_losses'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_r2'], label='Validation R²')
plt.axhline(y=0.99, color='r', linestyle='--', label='Target R²=0.99')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()