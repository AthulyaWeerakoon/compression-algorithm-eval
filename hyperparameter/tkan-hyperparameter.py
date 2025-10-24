import sys
sys.path.append('..')

import wandb
import pandas as pd
import numpy as np
import torch
from kan.MultKAN import KAN
import compression_evaluation.methods as util_methods


def fill_missing_with_mean(series):
    """Fill missing values with the mean of adjacent non-null values."""
    nan_indices = series[series.isna() | series.isnull()].index
    for idx in nan_indices:
        prev_idx = idx - 1
        next_idx = idx + 1

        while prev_idx >= 0 and (pd.isna(series.iloc[prev_idx]) or pd.isnull(series.iloc[prev_idx])):
            prev_idx -= 1
        while next_idx < len(series) and (pd.isna(series.iloc[next_idx]) or pd.isnull(series.iloc[next_idx])):
            next_idx += 1

        if prev_idx >= 0 and next_idx < len(series):
            mean_val = (series.iloc[prev_idx] + series.iloc[next_idx]) / 2
            series.iloc[idx] = mean_val
    return series


# Device setup
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load and prepare dataset
air_temperature_df = pd.read_csv(r'../datasets/AriviyalN_Data_Air_Temperature.csv')
air_temperature_column = air_temperature_df['Air Temperature']

# Clean data
air_temperature_column = fill_missing_with_mean(air_temperature_column)

# Split data: 70% train, 20% validation, 10% test
train_split = int(len(air_temperature_column) * 0.7)
valid_split = int(len(air_temperature_column) * 0.9)

air_temp_train = air_temperature_column[:train_split]
air_temp_valid = air_temperature_column[train_split:valid_split]
air_temp_test = air_temperature_column[valid_split:]

print(f"\nDataset splits:")
print(f"Training: {len(air_temp_train)} samples")
print(f"Validation: {len(air_temp_valid)} samples")
print(f"Testing: {len(air_temp_test)} samples")


def build_kan_width(config):
    """Build KAN width list dynamically based on config."""
    width = [config.input_steps]  # variable input size

    if config.num_hidden_layers >= 1 and hasattr(config, 'hidden_layer1'):
        if config.hidden_layer1 > 0:
            width.append(config.hidden_layer1)
    if config.num_hidden_layers == 2 and hasattr(config, 'hidden_layer2'):
        if config.hidden_layer2 > 0:
            width.append(config.hidden_layer2)

    width.append(config.output_steps)  # variable output size
    return width


def train_kan_model():
    """Train KAN model with WandB logging."""
    wandb.init(project="kan-air-temperature-hyperparameter")
    config = wandb.config

    # Prepare time series datasets with variable input/output
    X_train, y_train = util_methods.build_time_series_dataset(
        air_temp_train, config.input_steps, config.output_steps
    )
    X_valid, y_valid = util_methods.build_time_series_dataset(
        air_temp_valid, config.input_steps, config.output_steps, shuffle=False
    )

    X_train = torch.tensor(X_train, dtype=torch.float64, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float64, device=device)
    X_valid = torch.tensor(X_valid, dtype=torch.float64, device=device)
    y_valid = torch.tensor(y_valid, dtype=torch.float64, device=device)

    print(f"\nTraining shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shape: X={X_valid.shape}, y={y_valid.shape}")

    # Prepare dataset dictionary for KAN
    dataset = {
        'train_input': X_train,
        'train_label': y_train,
        'test_input': X_valid,
        'test_label': y_valid
    }

    # Build KAN width configuration
    width = build_kan_width(config)
    print(f"KAN width configuration: {width}")

    try:
        kan_model = KAN(
            width=width,
            grid=config.grid,
            k=config.k,
            seed=config.seed,
            device=device
        )

        kan_model.fit(
            dataset,
            steps=config.steps,
            lamb=config.lamb,
            lamb_entropy=config.lamb_entropy if hasattr(config, 'lamb_entropy') else 2.0
        )

        with torch.no_grad():
            y_train_pred = kan_model(X_train)
            y_valid_pred = kan_model(X_valid)

        train_residuals = (y_train - y_train_pred).cpu().numpy()
        valid_residuals = (y_valid - y_valid_pred).cpu().numpy()

        train_mse = np.mean(train_residuals ** 2)
        valid_mse = np.mean(valid_residuals ** 2)
        train_mae = np.mean(np.abs(train_residuals))
        valid_mae = np.mean(np.abs(valid_residuals))

        total_params = sum(p.numel() for p in kan_model.parameters())
        num_hidden_layers = len(width) - 2

        print(f"\nFinal Results:")
        print(f"Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Valid MSE: {valid_mse:.4f}, Valid MAE: {valid_mae:.4f}")
        print(f"Hidden layers: {num_hidden_layers}, Total params: {total_params}")

        wandb.log({
            "final_train_mse": train_mse,
            "final_valid_mse": valid_mse,
            "final_train_mae": train_mae,
            "final_valid_mae": valid_mae,
            "num_hidden_layers": num_hidden_layers,
            "total_params": total_params,
            "width_config": str(width)
        })

    except Exception as e:
        print(f"Error during training: {e}")
        wandb.log({"error": str(e), "status": "failed"})

    finally:
        wandb.finish()


# WandB Hyperparameter Sweep Configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'final_valid_mse',
        'goal': 'minimize'
    },
    'parameters': {
        # Variable input/output sizes
        'input_steps': {
            'values': [4, 6, 8, 10]
        },
        'output_steps': {
            'values': [1, 2, 3, 4]
        },

        # Layer configuration
        'num_hidden_layers': {
            'values': [0, 1, 2]
        },
        'hidden_layer1': {
            'values': [3, 5, 8, 10, 16]
        },
        'hidden_layer2': {
            'values': [3, 5, 8, 10, 16]
        },

        # KAN-specific hyperparams
        'grid': {
            'values': [3, 5, 10, 20]
        },
        'k': {
            'values': [2, 3, 4, 5]
        },
        'steps': {
            'values': [50, 100, 200, 300]
        },
        'lamb': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
        'lamb_entropy': {
            'distribution': 'log_uniform_values',
            'min': 0.1,
            'max': 10.0
        },

        'seed': {
            'value': 42
        }
    }
}


if _name_ == "_main_":
    sweep_id = wandb.sweep(sweep_config, project="kan-air-temperature-hyperparameter")
    wandb.agent(sweep_id, function=train_kan_model, count=30)