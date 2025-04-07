import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
from datetime import datetime
from joblib import Parallel, parallel_backend,delayed
import joblib
from sklearn.feature_selection import SelectKBest, f_regression


import time
# === CONFIGURATION ET PRÉPARATION DES DOSSIERS DE RÉSULTATS ===
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_DL_results")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"{results_dir}_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

def log_message(message, log_file=os.path.join(results_dir, "training_log.txt")):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")

log_message("Starting model training process with enhanced features")

# === CHARGEMENT DES DONNÉES ===
Data_path = 'C:/Users/samad/OneDrive/Bureau/visulisation_project'
list_csv_files = [file for file in os.listdir(Data_path) if file.endswith('.csv')]
list_csv_files = sorted(list_csv_files)
liste_key_names = [os.path.splitext(name)[0] for name in list_csv_files]
dict_data = {
    key: pd.read_csv(os.path.join(Data_path, file), sep=';')
    for key, file in zip(tqdm(liste_key_names, desc="Loading data files"), list_csv_files)
}

X_train = dict_data['Dataset_numerique_20000_petites_machines']
X_test = dict_data['Dataset_numerique_10000_petites_machines']
y_train = X_train.pop('Cmoy')
y_test = X_test.pop('Cmoy')

# === ENHANCED FEATURE ENGINEERING ===
log_message("Starting feature engineering...")

# Step 1: Split data
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
log_message(f"Original feature count: {X_train_raw.shape[1]}")

# Step 2: Reduce dimensionality
selector = SelectKBest(score_func=f_regression, k=20)
X_train_raw_reduced = selector.fit_transform(X_train_raw, y_train_raw)
X_val_raw_reduced = selector.transform(X_val_raw)
X_test_reduced = selector.transform(X_test)

# Step 3: Fit polynomial features transformer
poly = PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
log_message("Fitting PolynomialFeatures on training data...")
X_train_poly = poly.fit_transform(X_train_raw_reduced)

# Step 4: Define parallel transform function
def timed_transform(name, data):
    start = time.time()
    transformed = poly.transform(data)
    duration = time.time() - start
    return name, transformed, duration

# Step 5: Parallel transform on val and test sets
data_dict = {
    "val": X_val_raw_reduced,
    "test": X_test_reduced
}

log_message("Transforming validation and test data in parallel...")
results = Parallel(n_jobs=-1)(
    delayed(timed_transform)(name, data) for name, data in tqdm(data_dict.items(), desc="Poly transform", ncols=100)
)

# Step 6: Unpack results
for name, transformed, duration in results:
    if name == "val":
        X_val_poly = transformed
    elif name == "test":
        X_test_poly = transformed
    log_message(f"{name} poly shape: {transformed.shape}, time: {duration:.2f}s")

log_message(f"Features after polynomial transformation: {X_train_poly.shape[1]}")

# Step 7: Save durations to metrics.json
metrics_path = "metrics1.json"
metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

metrics["train_poly_duration"] = 0  # already done during fit_transform
for name, _, duration in results:
    metrics[f"{name}_poly_duration"] = round(duration, 2)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# Step 3: Feature selection using Random Forest importance
log_message("Performing feature selection with Random Forest...")


# Patch pour tqdm + joblib
from tqdm.auto import tqdm as tqdm_auto
import joblib.parallel


class TqdmJoblibProgressBar(tqdm_auto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ProgressParallel(joblib.Parallel):
    def __init__(self, tqdm_bar, *args, **kwargs):
        self._tqdm_bar = tqdm_bar
        super().__init__(*args, **kwargs)

    def print_progress(self):
        if self._tqdm_bar is not None:
            completed = self.n_completed_tasks
            total = self.n_dispatched_tasks
            self._tqdm_bar.total = total
            self._tqdm_bar.n = completed
            self._tqdm_bar.refresh()
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

with TqdmJoblibProgressBar(desc="Fitting RandomForest", total=100) as progress_bar:
    with parallel_backend('threading'):
        with ProgressParallel(progress_bar) as parallel:
            rf_selector.set_params(n_jobs=parallel.n_jobs)
            rf_selector.fit(X_train_poly, y_train_raw)

# Get feature importances and select the top features
feature_selector = SelectFromModel(rf_selector, threshold="median")
with tqdm(total=3, desc="Feature selection") as pbar:
    X_train_selected = feature_selector.fit_transform(X_train_poly, y_train_raw)
    pbar.update(1)
    X_val_selected = feature_selector.transform(X_val_poly)
    pbar.update(1)
    X_test_selected = feature_selector.transform(X_test_poly)
    pbar.update(1)

log_message(f"Features after selection: {X_train_selected.shape[1]}")

# Save selected feature indices
selected_indices = feature_selector.get_support(indices=True)
with open(os.path.join(results_dir, "selected_features.json"), "w") as f:
    json.dump({"selected_indices": selected_indices.tolist()}, f)

# Standardization
scaler = StandardScaler()
with tqdm(total=3, desc="Standardizing data") as pbar:
    X_train_scaled = scaler.fit_transform(X_train_selected)
    pbar.update(1)
    X_val_scaled = scaler.transform(X_val_selected)
    pbar.update(1)
    X_test_scaled = scaler.transform(X_test_selected)
    pbar.update(1)

# Tensors
with tqdm(total=5, desc="Creating tensors") as pbar:
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    pbar.update(1)
    y_train_tensor = torch.tensor(y_train_raw.values.reshape(-1, 1), dtype=torch.float32)
    pbar.update(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    pbar.update(1)
    y_val_tensor = torch.tensor(y_val_raw.values.reshape(-1, 1), dtype=torch.float32)
    pbar.update(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_np = y_test.values.reshape(-1, 1)
    pbar.update(1)

# Loaders
batch_size = 256  # Increased batch size
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

# === ADVANCED MODEL WITH RESIDUAL CONNECTIONS ===
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU(0.1)
        
        # Skip connection if dimensions don't match
        self.skip = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.activation(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += identity  # Skip connection
        return self.activation(out)

class EnhancedResidualMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Initial layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(1024, 512)
        self.res_block2 = ResidualBlock(512, 256)
        self.res_block3 = ResidualBlock(256, 128)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.output_layers(x)

# === LOSS FUNCTIONS ===
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)

# Combined loss function (MSE + L1)
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, y_pred, y_true):
        return self.alpha * self.mse(y_pred, y_true) + (1 - self.alpha) * self.mae(y_pred, y_true)

# === HYPERPARAMETER TUNING WITH CROSS-VALIDATION ===
def cross_validate(input_size, n_splits=5, learning_rates=[1e-3, 5e-4, 1e-4]):
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_val_score = -float('inf')
    best_params = {}
    
    # Convert to numpy for KFold
    X_train_np = X_train_scaled
    y_train_np = y_train_raw.values
    
    log_message("Starting cross-validation for hyperparameter tuning...")
    
    with tqdm(learning_rates, desc="Cross-validation") as lr_pbar:
        for lr in lr_pbar:
            cv_scores = []
            
            fold_pbar = tqdm(enumerate(k_fold.split(X_train_np)), total=n_splits, 
                            desc=f"LR={lr}", leave=False)
            for fold, (train_idx, val_idx) in fold_pbar:
                X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
                y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]
                
                # Convert to tensors
                X_fold_train_tensor = torch.tensor(X_fold_train, dtype=torch.float32)
                y_fold_train_tensor = torch.tensor(y_fold_train.reshape(-1, 1), dtype=torch.float32)
                X_fold_val_tensor = torch.tensor(X_fold_val, dtype=torch.float32)
                y_fold_val_tensor = torch.tensor(y_fold_val.reshape(-1, 1), dtype=torch.float32)
                
                # Create dataloaders
                fold_train_loader = DataLoader(TensorDataset(X_fold_train_tensor, y_fold_train_tensor), 
                                            batch_size=batch_size, shuffle=True)
                fold_val_loader = DataLoader(TensorDataset(X_fold_val_tensor, y_fold_val_tensor), 
                                            batch_size=batch_size)
                
                # Create and train model
                model = EnhancedResidualMLP(input_size)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
                criterion = CombinedLoss()
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
                
                # Mini training
                for epoch in range(30):  # Short training for CV
                    model.train()
                    for X_batch, y_batch in fold_train_loader:
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                    fold_pbar.set_postfix({"epoch": f"{epoch+1}/30"})
                
                # Evaluate
                model.eval()
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for X_batch, y_batch in fold_val_loader:
                        preds = model(X_batch)
                        all_preds.append(preds.numpy())
                        all_targets.append(y_batch.numpy())
                
                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)
                fold_r2 = r2_score(all_targets, all_preds)
                cv_scores.append(fold_r2)
                
                fold_pbar.set_postfix({"R²": f"{fold_r2:.4f}"})
                log_message(f"LR={lr}, Fold {fold+1}/{n_splits}: R² = {fold_r2:.4f}")
            
            avg_score = np.mean(cv_scores)
            lr_pbar.set_postfix({"Avg R²": f"{avg_score:.4f}"})
            log_message(f"LR={lr} average R² = {avg_score:.4f}")
            
            if avg_score > best_val_score:
                best_val_score = avg_score
                best_params['learning_rate'] = lr
    
    log_message(f"Best parameters from CV: {best_params}")
    return best_params

# === TRAINING WITH EARLY STOPPING AND SCHEDULER ===
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=300, patience=30, model_index=0):
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0
    metrics_log = {
        "train_loss": [],
        "val_loss": [],
        "val_r2": [],
        "learning_rates": []
    }
    
    # Save initial model state
    best_model_state = model.state_dict().copy()

    with tqdm(range(epochs), desc=f"Model {model_index+1} Training") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False) as t:
                for X_batch, y_batch in t:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)
                    
                    # Update progress bar
                    t.set_postfix(loss=f"{loss.item():.4f}")
            
            train_loss /= len(train_loader.dataset)

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val] ", leave=False) as t:
                for X_batch, y_batch in t:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    
                    # Collect predictions for R² calculation
                    all_preds.append(outputs.numpy())
                    all_targets.append(y_batch.numpy())
                    
                    # Update progress bar
                    t.set_postfix(loss=f"{loss.item():.4f}")
            
            val_loss /= len(val_loader.dataset)
            
            # Calculate R² for validation set
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            val_r2 = r2_score(all_targets, all_preds)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            metrics_log["train_loss"].append(train_loss)
            metrics_log["val_loss"].append(val_loss)
            metrics_log["val_r2"].append(val_r2)
            metrics_log["learning_rates"].append(current_lr)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}", 
                "val_loss": f"{val_loss:.4f}", 
                "R²": f"{val_r2:.4f}",
                "patience": f"{patience_counter}/{patience}"
            })
            
            # Print progress
            log_message(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R²: {val_r2:.4f}, LR: {current_lr:.6f}")
            
            # Update scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Early stopping based on R² score
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
                log_message(f"New best model with R² = {val_r2:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log_message(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Save metrics for this model
    with open(os.path.join(results_dir, f"metrics_model_{model_index+1}.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)
    
    log_message(f"Model {model_index+1} finished training with best R² = {best_r2:.4f}")
    return model, best_r2

def create_ensemble(input_size, num_models=10):
    return [EnhancedResidualMLP(input_size) for _ in range(num_models)]

def train_ensemble(models, train_loader, val_loader, best_params, epochs=300, patience=30):
    trained_models = []
    best_r2_scores = []
    criteria = [
        nn.MSELoss(),
        CombinedLoss(alpha=0.8),
        CombinedLoss(alpha=0.5),
        HuberLoss(delta=1.0),
        HuberLoss(delta=0.5),
    ]
    
    # Ensure we have enough criteria or cycle through them
    if len(criteria) < len(models):
        criteria = criteria * (len(models) // len(criteria) + 1)
    
    with tqdm(enumerate(models), total=len(models), desc="Training ensemble") as ensemble_pbar:
        for idx, model in ensemble_pbar:
            ensemble_pbar.set_description(f"Training model {idx + 1}/{len(models)}")
            log_message(f"Training model {idx + 1}/{len(models)}")
            
            # Use different learning rates and optimizers for diversity
            lr = best_params.get('learning_rate', 1e-3)
            if idx % 3 == 0:
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
                log_message(f"Using AdamW with CosineAnnealingWarmRestarts")
            elif idx % 3 == 1:
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
                log_message(f"Using Adam with ReduceLROnPlateau")
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr*10, momentum=0.9, weight_decay=1e-4)
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                log_message(f"Using SGD with CosineAnnealingWarmRestarts")
            
            # Use different loss functions for diversity
            criterion = criteria[idx % len(criteria)]
            criterion_name = criterion.__class__.__name__
            log_message(f"Using {criterion_name} loss function")
            
            # Train model
            trained_model, best_r2 = train_model(
                model, criterion, optimizer, scheduler,
                train_loader, val_loader, epochs, patience,
                model_index=idx
            )
            
            trained_models.append(trained_model)
            best_r2_scores.append(best_r2)
            ensemble_pbar.set_postfix({"R²": f"{best_r2:.4f}"})
    
    # Log best models
    best_models_indices = np.argsort(best_r2_scores)[::-1]
    log_message(f"Best model indices by R² score: {best_models_indices.tolist()}")
    
    return [trained_models[i] for i in best_models_indices]

def ensemble_predict(models, X_tensor, top_k=None):
    """Make predictions with ensemble, optionally using only top-k models"""
    if top_k is not None and top_k < len(models):
        models = models[:top_k]  # Take only top-k models
    
    all_preds = []
    with tqdm(models, desc="Ensemble prediction", total=len(models)) as pred_pbar:
        with torch.no_grad():
            for model in pred_pbar:
                model.eval()
                pred = model(X_tensor).numpy()
                all_preds.append(pred)
    
    # Calculate ensemble prediction (average)
    ensemble_pred = np.mean(all_preds, axis=0)
    
    # Calculate predictions variance (model uncertainty)
    pred_variance = np.var(all_preds, axis=0)
    
    return ensemble_pred, pred_variance

# === MAIN PIPELINE ===
# Step 1: Cross-validation for hyperparameter tuning
input_size = X_train_scaled.shape[1]
best_params = cross_validate(input_size)

# Step 2: Train ensemble with best parameters
num_models = 10  # Increased number of models
log_message(f"Training ensemble of {num_models} models with best parameters")
ensemble_models = train_ensemble(
    create_ensemble(input_size, num_models),
    train_loader, 
    val_loader,
    best_params,
    epochs=300,
    patience=30
)

# Step 3: Make predictions with ensemble
log_message("Making predictions with full ensemble")
preds, pred_variance = ensemble_predict(ensemble_models, X_test_tensor)

# Step 4: Evaluate full ensemble
r2_full = r2_score(y_test_np, preds)
mse_full = np.mean((y_test_np - preds) ** 2)
mae_full = np.mean(np.abs(y_test_np - preds))

results_message = f"""
Full Ensemble Performance ({num_models} models):
R² Score: {r2_full:.4f}
MSE: {mse_full:.4f}
MAE: {mae_full:.4f}
"""
log_message(results_message)

# Step 5: Try different ensemble sizes
with tqdm(total=3, desc="Evaluating ensemble sizes") as size_pbar:
    for k in [3, 5, 7]:
        if k < len(ensemble_models):
            size_pbar.set_description(f"Evaluating top-{k} models")
            log_message(f"Evaluating top-{k} models ensemble")
            preds_k, _ = ensemble_predict(ensemble_models, X_test_tensor, top_k=k)
            r2_k = r2_score(y_test_np, preds_k)
            log_message(f"Top-{k} models ensemble R²: {r2_k:.4f}")
            size_pbar.update(1)

# Step 6: Save final results
with open(os.path.join(results_dir, "final_metrics.json"), "w") as f:
    json.dump({
        "R2_full": r2_full,
        "MSE_full": mse_full,
        "MAE_full": mae_full,
        "model_count": num_models
    }, f, indent=4)

log_message("Training pipeline completed")