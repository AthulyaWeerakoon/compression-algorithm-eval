import wandb
import pandas as pd
import torch
from kan.MultKAN import KAN
import compression_evaluation.methods as util_methods

# ---------------------------
# Initialize W&B Sweep run
# ---------------------------
wandb.init(project="tkan")
config = wandb.config

# ---------------------------
# Device setup
# ---------------------------
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ---------------------------
# Prepare dataset (example)
# ---------------------------
air_temperature_df = pd.read_csv(r'..\datasets\AriviyalN_Data_Air_Temperature.csv')
air_temperature_column = air_temperature_df['Air Temperature']
# split training dataset into training and validating datasets
air_temperature_training = air_temperature_column[:(len(air_temperature_column) * 9) // 10]
air_temperature_testing = air_temperature_column[(len(air_temperature_column) * 9) // 10:]
air_temperature_validating = air_temperature_training[(len(air_temperature_column) * 7) // 9:]
air_temperature_training = air_temperature_training[:(len(air_temperature_column) * 7) // 9]


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

# ---------------------------
# Build KAN width list dynamically
# ---------------------------
width = [8]  # input size fixed
for hl in [config.hidden_layer1, config.hidden_layer2, config.hidden_layer3]:
    if hl > 0:
        width.append(hl)
width.append(3)  # output size fixed

print("KAN width configuration:", width)

# ---------------------------
# Initialize and train KAN
# ---------------------------
kan_model = KAN(
    width=width,
    grid=config.KAN_grid,
    k=config.KAN_k,
    seed=config.random_seed,
    device=device
)

kan_model.fit(dataset)

# ---------------------------
# Evaluate
# ---------------------------
y_valid_pred = kan_model(dataset['test_input'])
residuals = (dataset['test_label'] - y_valid_pred).cpu().detach().numpy()
mse_valid = (residuals**2).mean()
print("Validation MSE:", mse_valid)

wandb.log({"mse_valid": mse_valid})
wandb.finish()
