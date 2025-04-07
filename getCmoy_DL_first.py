
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.neural_network import MLPRegressor
#################################################################################
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#################################################################################
Data_path = 'C:/Users/samad/OneDrive/Bureau/visulisation_project'
#Liste des fichier .csv dans le repertoire
list_csv_files = []
for n_dir, _, n_files in os.walk(Data_path):
   list_csv_files = [file for file in n_files if file.endswith('.csv')]
list_csv_files = sorted(list_csv_files)
liste_key_names = [os.path.splitext(name)[0][:] for name in list_csv_files]
dict_data = {}
for k, n_file in zip(liste_key_names, list_csv_files):
    print(k,'---', n_file)
    dict_data[k] = pd.read_csv(Data_path + '/' + n_file,sep=';') 
"""
Dataset_analytique_10000_petites_machines --- Dataset_analytique_10000_petites_machines.csv
Dataset_analytique_20000_petites_machines --- Dataset_analytique_20000_petites_machines.csv
Dataset_numerique_10000_petites_machines --- Dataset_numerique_10000_petites_machines.csv
Dataset_numerique_20000_petites_machines --- Dataset_numerique_20000_petites_machines.csv

"""
X_train = dict_data['Dataset_numerique_20000_petites_machines']
X_test = dict_data['Dataset_numerique_10000_petites_machines']
y_train = X_train.pop('Cmoy')
y_test = X_test.pop('Cmoy')

'''
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # 3 couches
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    random_state=42
)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Pas d'activation pour une sortie continue
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
'''
class RegressionMLP(nn.Module):
    def __init__(self, input_size):
        super(RegressionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1) # sortie continue
        )

    def forward(self, x):
        return self.model(x)
        
def train_model(model, criterion, optimizer, X_train, y_train, epochs=200):
    for epoch in range(epochs):
        model.train()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Époque {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            
# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversion en tenseurs
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Modèle
input_size = X_train.shape[1]
model = RegressionMLP(input_size)

# Perte et optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement
train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, epochs=300)

# Évaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

r2 = r2_score(y_test, y_pred)
print(f"Score R² sur les données test : {r2:.4f}")