# KAN Predictive Coding with W&B
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from kan.MultKAN import KAN
import compression_evaluation.methods as util_methods
import compression_evaluation.classes as util_classes
import compression_evaluation.types as util_types
import compression_evaluation.evaluator as util_evaluator

# 1. W&B sweep config
# This assumes you run it with a W&B sweep, so the agent injects hyperparameters
config = wandb.config

# Example sweep.yaml should contain:
# KAN_width: [8, 5, 3]
# KAN_grid: 1
# KAN_k: 1
# random_seed: 42
# bin_size: 0.1

# 2. Load dataset
air_temperature_df = pd.read_csv(r'..\datasets\AriviyalN_Data_Air_Temperature.csv')
air_temperature_column = air_temperature_df['Air Temperature']

# Fill missing values
def fill_with_mean(series):
    nan_indices = series[series.isna() | series.isnull()].index
    for idx in nan_indices:
        prev_idx, next_idx = idx - 1, idx + 1
        while prev_idx >= 0 and (pd.isna(series.iloc[prev_idx]) or pd.isnull(series.iloc[prev_idx])):
            prev_idx -= 1
        while next_idx < len(series) and (pd.isna(series.iloc[next_idx]) or pd.isnull(series.iloc[next_idx])):
            next_idx += 1
        if prev_idx >= 0 and next_idx < len(series):
            series.iloc[idx] = (series.iloc[prev_idx] + series.iloc[next_idx]) / 2
    return series

air_temperature_column = fill_with_mean(air_temperature_column)

# 3. Split dataset
train_ratio = 0.7
valid_ratio = 0.2  
test_ratio = 0.1

n = len(air_temperature_column)
train_end = int(n * train_ratio)
valid_end = train_end + int(n * valid_ratio)

air_temperature_training = air_temperature_column[:train_end]
air_temperature_validating = air_temperature_column[train_end:valid_end]
air_temperature_testing = air_temperature_column[valid_end:]

# 4. Prepare tensors
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

X_train, y_train = util_methods.build_time_series_dataset(air_temperature_training, 8, 3)
X_valid, y_valid = util_methods.build_time_series_dataset(air_temperature_validating, 8, 3, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float64, device=device)
y_train = torch.tensor(y_train, dtype=torch.float64, device=device)

X_valid = torch.tensor(X_valid, dtype=torch.float64, device=device)
y_valid = torch.tensor(y_valid, dtype=torch.float64, device=device)

dataset = {
    'train_input': X_train,
    'train_label': y_train,
    'test_input': X_valid,
    'test_label': y_valid
}

# 5. Initialize & train KAN
kan_model = KAN(
    width=config.KAN_width,
    grid=config.KAN_grid,
    k=config.KAN_k,
    seed=config.random_seed,
    device=device
)

kan_model.fit(dataset)

# 6. Predictions
y_train_pred = kan_model(dataset['train_input'])
y_valid_pred = kan_model(dataset['test_input'])

# 7. Compute residuals & MSE
residuals_train = (dataset['train_label'] - y_train_pred).detach().cpu().numpy()
residuals_valid = (dataset['test_label'] - y_valid_pred).detach().cpu().numpy()

mse_train = np.mean(residuals_train**2)
mse_valid = np.mean(residuals_valid**2)

print(f"Train MSE: {mse_train:.4f}")
print(f"Validation MSE: {mse_valid:.4f}")

# 8. W&B logging
wandb.log({
    "mse_train": mse_train,
    "mse_valid": mse_valid,
    "train_residuals": wandb.Histogram(residuals_train),
    "valid_residuals": wandb.Histogram(residuals_valid)
})

# 9. Residual histogram plot 
plt.figure(figsize=(8, 5))
plt.hist(residuals_train, bins=np.arange(residuals_train.min(), residuals_train.max() + config.bin_size, config.bin_size),
         alpha=0.7, label="Train Residuals")
plt.hist(residuals_valid, bins=np.arange(residuals_valid.min(), residuals_valid.max() + config.bin_size, config.bin_size),
         alpha=0.7, label="Valid Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
