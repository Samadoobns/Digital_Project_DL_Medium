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

log_message("Starting model training process")

# === CHARGEMENT DES DONNÉES ===
Data_path = 'C:/Users/samad/OneDrive/Bureau/visulisation_project'
list_csv_files = [file for file in os.listdir(Data_path) if file.endswith('.csv')]
list_csv_files = sorted(list_csv_files)
liste_key_names = [os.path.splitext(name)[0] for name in list_csv_files]
dict_data = {
    key: pd.read_csv(os.path.join(Data_path, file), sep=';')
    for key, file in zip(liste_key_names, list_csv_files)
}

X_train = dict_data['Dataset_numerique_20000_petites_machines']
X_test = dict_data['Dataset_numerique_10000_petites_machines']
y_train = X_train.pop('Cmoy')
y_test = X_test.pop('Cmoy')

# === SPLIT ET PRÉTRAITEMENTS ===
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = pd.DataFrame(poly.fit_transform(X_train_raw))
X_val_poly = pd.DataFrame(poly.transform(X_val_raw))
X_test_poly = pd.DataFrame(poly.transform(X_test))

# Sauvegarde des colonnes
with open(os.path.join(results_dir, "feature_names.json"), "w") as f:
    json.dump(list(X_train_poly.columns), f)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_val_scaled = scaler.transform(X_val_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw.values.reshape(-1, 1), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_raw.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_np = y_test.values.reshape(-1, 1)

# Loaders
batch_size = 128
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

# === MODÈLE ===
class EnhancedRegressionMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024), nn.LeakyReLU(0.1), nn.BatchNorm1d(1024), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.1), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.1), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.1), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.BatchNorm1d(64), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.LeakyReLU(0.1), nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

# === ENTRAÎNEMENT ===
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=200, patience=20, model_index=0):
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_log = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Model {model_index+1} - Epoch {epoch+1}/{epochs} [Train]", leave=False) as t:
            for X_batch, y_batch in t:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), tqdm(val_loader, desc=f"Model {model_index+1} - Epoch {epoch+1}/{epochs} [Val] ", leave=False) as t:
            for X_batch, y_batch in t:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        metrics_log["train_loss"].append(train_loss)
        metrics_log["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Sauvegarde des métriques pour ce modèle
    with open(os.path.join(results_dir, f"metrics_model_{model_index+1}.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)

    return model

def create_ensemble(input_size, num_models=5):
    return [EnhancedRegressionMLP(input_size) for _ in range(num_models)]

def train_ensemble(models, criterion, train_loader, val_loader, epochs=200, patience=20):
    trained_models = []
    for idx, model in enumerate(models):
        log_message(f"Training model {idx + 1}/{len(models)}")
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        trained_model = train_model(
            model, criterion, optimizer, scheduler,
            train_loader, val_loader, epochs, patience,
            model_index=idx
        )
        trained_models.append(trained_model)
    return trained_models
def ensemble_predict(models, X_tensor):
    all_preds = []
    with torch.no_grad():
        for model in models:
            model.eval()
            pred = model(X_tensor).numpy()
            all_preds.append(pred)
    return np.mean(all_preds, axis=0)

# === LANCEMENT DU PIPELINE ===
input_size = X_train_scaled.shape[1]
ensemble_models = train_ensemble(
    create_ensemble(input_size),
    criterion=nn.MSELoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
    patience=15
)
preds = ensemble_predict(ensemble_models, X_test_tensor)

# === ÉVALUATION ===
r2 = r2_score(y_test_np, preds)
mse = np.mean((y_test_np - preds) ** 2)
mae = np.mean(np.abs(y_test_np - preds))

results_message = f"""
Final Model Performance:
R² Score: {r2:.4f}
MSE: {mse:.4f}
MAE: {mae:.4f}
"""
print(results_message)
log_message(results_message)

with open(os.path.join(results_dir, "final_metrics.json"), "w") as f:
    json.dump({"R2": r2, "MSE": mse, "MAE": mae}, f, indent=4)
