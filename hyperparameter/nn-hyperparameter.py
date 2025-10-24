import sys
sys.path.append('..')

import wandb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import compression_evaluation.methods as util_methods


# Load data
air_temperature_df = pd.read_csv(r'..\datasets\AriviyalN_Data_Air_Temperature.csv')
air_temperature_column = air_temperature_df['Air Temperature']


def print_invalid_counts():
    """Print count of NaN and null values in the data."""
    print("Air Temperature NaNs:", air_temperature_column.isna().sum())
    print("Air Temperature Nulls:", air_temperature_column.isnull().sum())


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


def build_model(input_steps, output_steps, num_hidden_layers, hidden_units, learning_rate, activation='relu'):
    """Build neural network model with variable architecture."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_steps,)))

    # Optionally add 1–2 hidden layers
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(hidden_units, activation=activation))

    # Always end with output layer
    model.add(layers.Dense(output_steps))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mse', 'mae']
    )
    return model


class WandbMetricsLogger(keras.callbacks.Callback):
    """Custom callback to log metrics to WandB."""
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log({
                'epoch': epoch,
                'loss': logs.get('loss'),
                'mse': logs.get('mse'),
                'mae': logs.get('mae'),
                'val_loss': logs.get('val_loss'),
                'val_mse': logs.get('val_mse'),
                'val_mae': logs.get('val_mae')
            })


def train_model():
    """Train model with WandB logging."""
    wandb.init(project="nn-air-temperature-hyperparameter")
    config = wandb.config

    # Prepare time series datasets with variable window sizes
    X_train, y_train = util_methods.build_time_series_dataset(
        air_temp_train, config.input_steps, config.output_steps
    )
    X_valid, y_valid = util_methods.build_time_series_dataset(
        air_temp_valid, config.input_steps, config.output_steps, shuffle=False
    )

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_valid = X_valid.astype(np.float32)
    y_valid = y_valid.astype(np.float32)

    print(f"\nTraining shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shape: X={X_valid.shape}, y={y_valid.shape}")

    # Build model with current sweep parameters
    model = build_model(
        input_steps=config.input_steps,
        output_steps=config.output_steps,
        num_hidden_layers=config.num_hidden_layers,
        hidden_units=config.hidden_units,
        learning_rate=config.learning_rate,
        activation=config.activation
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            WandbMetricsLogger()
        ]
    )

    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)

    train_mse = np.mean((y_train - y_train_pred) ** 2)
    valid_mse = np.mean((y_valid - y_valid_pred) ** 2)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    valid_mae = np.mean(np.abs(y_valid - y_valid_pred))

    print(f"\nFinal Results:")
    print(f"Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}")
    print(f"Valid MSE: {valid_mse:.4f}, Valid MAE: {valid_mae:.4f}")

    wandb.log({
        "final_train_mse": train_mse,
        "final_valid_mse": valid_mse,
        "final_train_mae": train_mae,
        "final_valid_mae": valid_mae
    })

    wandb.finish()


sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'input_steps': {
            'values': [4, 6, 8, 10]
        },
        'output_steps': {
            'values': [1, 2, 3, 4]
        },
        'num_hidden_layers': {
            'values': [0, 1, 2]
        },
        'hidden_units': {
            'values': [3, 5, 8, 10, 16]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
        'batch_size': {
            'values': [8, 16, 32, 64]
        },
        'epochs': {
            'value': 100
        },
        'activation': {
            'values': ['relu', 'tanh', 'elu']
        }
    }
}


if _name_ == "_main_":
    sweep_id = wandb.sweep(sweep_config, project="nn-air-temperature-hyperparameter")
    wandb.agent(sweep_id, function=train_model, count=20)