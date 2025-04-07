🧠 Deep Learning Regression Pipeline for Torque Prediction - File Structure
Below is an organized breakdown of each Python file in the project and its specific role:
File Structure Overview
1. getCmoy_DL_first.py
Basic PyTorch MLP Implementation

Implements a basic deep feedforward neural network (MLP) using PyTorch
Architecture: 512 → 256 → 128 → 64 → 32 → 1 with ReLU activations
Trains for 300 epochs using Adam optimizer and MSE loss
Loads data from CSV files and performs standard scaling
Provides initial R² evaluation on test data

2. getCmoy_DL_second.py
Advanced MLP with Regularization

Enhances the basic MLP with batch normalization and dropout
Architecture includes dropout layers (0.1-0.3) after each hidden layer
Implements early stopping with patience=15 and ReduceLROnPlateau scheduler
Includes training history visualization and residual analysis
Creates comprehensive output including metrics, plots, and saved model

3. getCmoy_DL_fourth.py
Feature Engineering Pipeline

Adds polynomial feature expansion (degree 2)
Implements a deep neural network with leaky ReLU activations
Includes data imputation for handling missing values
Provides detailed training logs with timestamp tracking
Focuses on performance visualization with multiple metrics (R², MSE, MAE)

4. getCmoy_DL_fifth.py
Ensemble Learning Implementation

Creates an ensemble of 5 models for more robust predictions
Implements custom architecture with BatchNorm, Dropout, and LeakyReLU
Organizes outputs in time-stamped results folders
Stores metrics for each individual model and final ensemble performance
Provides automatic logging and feature name tracking

5. getCmoy_DL_sixth.py
Advanced Residual Network with Custom Loss

Implements residual connections for better gradient flow in deep networks
Creates EnhancedResidualMLP with Layer Normalization for stability
Defines custom CombinedLoss function (weighted MSE + MAE)
Implements diverse ensemble with multiple loss functions and optimizers
Features model diversity strategies for improved ensemble performance
Provides uncertainty estimation through prediction variance

Integration Points
These files represent an evolution of approaches, from:

Basic implementation →
Adding regularization →
Feature engineering →
Ensemble methods →
Advanced architectures

The complete pipeline combines the strengths of each approach, allowing for:

Robust torque prediction with quantified uncertainty
Comprehensive evaluation metrics and visualizations
Production-ready model deployment with inference capabilities
Detailed logging and performance tracking
