# 🧠 Deep Learning Regression Pipeline for Torque Prediction

This project implements a comprehensive deep learning solution to predict electromagnetic torque (Cmoy) in electric motors using PyTorch.

---

## 📋 File Structure Overview

### 1. `getCmoy_DL_first.py` - 🔰 Basic PyTorch MLP Implementation
- Implements a basic deep feedforward neural network (MLP) using PyTorch  
- **Architecture**: 512 → 256 → 128 → 64 → 32 → 1 with ReLU activations  
- Trains for 300 epochs using Adam optimizer and MSE loss  
- Loads data from CSV files and performs standard scaling  
- Provides initial R² evaluation on test data  

### 2. `getCmoy_DL_second.py` - 🛡️ Advanced MLP with Regularization
- Enhances the basic MLP with batch normalization and dropout  
- **Architecture includes dropout layers (0.1–0.3)** after each hidden layer  
- Implements **early stopping** with `patience=15` and `ReduceLROnPlateau` scheduler  
- Includes **training history visualization and residual analysis**  
- Creates comprehensive output including metrics, plots, and saved model  

### 3. `getCmoy_DL_fourth.py` - 🧩 Feature Engineering Pipeline
- Adds **polynomial feature expansion** (degree 2)  
- Implements a deep neural network with **Leaky ReLU** activations  
- Includes **data imputation** for handling missing values  
- Provides detailed training logs with **timestamp tracking**  
- Focuses on performance visualization with multiple metrics (R², MSE, MAE)  

### 4. `getCmoy_DL_fifth.py` - 🤝 Ensemble Learning Implementation
- Creates an **ensemble of 5 models** for more robust predictions  
- Implements **custom architecture** with BatchNorm, Dropout, and LeakyReLU  
- Organizes outputs in **time-stamped results folders**  
- Stores metrics for each individual model and final ensemble performance  
- Provides **automatic logging and feature name tracking**  

### 5. `getCmoy_DL_sixth.py` - 🌉 Advanced Residual Network with Custom Loss
- Implements **residual connections** for better gradient flow in deep networks  
- Creates `EnhancedResidualMLP` with **Layer Normalization** for stability  
- Defines custom `CombinedLoss` function (**weighted MSE + MAE**)  
- Implements **diverse ensemble** with multiple loss functions and optimizers  
- Features **model diversity strategies** for improved ensemble performance  
- Provides **uncertainty estimation** through prediction variance  

---
## 📊 Model Architecture Comparison

| File | Architecture | Regularization | Loss Function | Special Features |
|------|--------------|----------------|---------------|-----------------|
| `getCmoy_DL_first.py` | Basic MLP (512→256→128→64→32→1) | None | MSE | Standard scaling |
| `getCmoy_DL_second.py` | MLP with BN & Dropout | Dropout 0.1-0.3, BatchNorm | MSE | Early stopping, LR scheduling |
| `getCmoy_DL_fourth.py` | Deep NN with LeakyReLU | Dropout | MSE | Polynomial features |
| `getCmoy_DL_fifth.py` | Ensemble of 5 models | Dropout, BatchNorm | MSE | Model averaging |
| `getCmoy_DL_sixth.py` | Residual MLP | LayerNorm | Combined (MSE+MAE) | Uncertainty estimation |
## 🔄 Integration Evolution

```mermaid
graph TD
    A[Basic MLP<br>getCmoy_DL_first.py] --> B[Advanced MLP<br>getCmoy_DL_second.py]
    B --> C[Feature Engineering<br>getCmoy_DL_fourth.py]
    C --> D[Ensemble Learning<br>getCmoy_DL_fifth.py]
    D --> E[Residual Networks<br>getCmoy_DL_sixth.py]

    style A fill:#f9d5e5,stroke:#333,stroke-width:2px
    style B fill:#eeac99,stroke:#333,stroke-width:2px
    style C fill:#e06377,stroke:#333,stroke-width:2px
    style D fill:#c83349,stroke:#333,stroke-width:2px
    style E fill:#5b9aa0,stroke:#333,stroke-width:2px

