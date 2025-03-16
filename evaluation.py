import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Training import TradingEnv, MacroAgent, MicroAgent  # Import necessary classes

# Load test dataset
stock=input("Enter Ticker to evaluate on:")
data_path = f"./Dataset/Evaluation/labeled_{stock}.NS.csv"
df = pd.read_csv(data_path)

# Ensure correct window size
window_size = 10  # Must match training

# Select relevant features (same as training)
feature_columns = ['volume', 'open', 'high', 'low', 'close', 'SMA_10', 'EMA_10', 'RSI_14',
                   'MACD', 'High-Low', 'High-Close', 'Low-Close', 'True Range', 'ATR', 
                   'STD_20', 'Upper_Band', 'Lower_Band', 'OBV', 'VWAP']

# Function to create windowed input features
def create_windowed_features(data, window_size):
    features = []
    for i in range(len(data) - window_size):
        window = data.iloc[i:i+window_size][feature_columns].values.flatten()  # Flatten window
        features.append(window)
    return np.array(features)

# Generate windowed test data
X_test = create_windowed_features(df, window_size)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load trained models
def load_model(path, model_class, input_dim):
    model = model_class(input_size=input_dim)
    model.load_state_dict(torch.load(path, weights_only=True))  # Use weights_only=True for security
    model.eval()
    return model

input_dim = X_test.shape[1]  # Ensure it matches training
macro_model = load_model("macro_model.pth", MacroAgent, input_dim)
micro_model_up = load_model("micro_model_up_trend.pth", MicroAgent, input_dim)
micro_model_sideways = load_model("micro_model_sideways.pth", MicroAgent, input_dim)
micro_model_down = load_model("micro_model_down_trend.pth", MicroAgent, input_dim)

# Verify input shape
print(f"Test data shape: {X_test_tensor.shape}")  # Should be (N, 190) if trained on 190 features

# Predict macro trend classification
macro_preds = macro_model(X_test_tensor).argmax(dim=1).numpy()

# Map predictions to micro models
final_actions = []
for i, trend in enumerate(macro_preds):
    if trend == 0:
        action = micro_model_down(X_test_tensor[i].unsqueeze(0)).detach().numpy()
    elif trend == 1:
        action = micro_model_sideways(X_test_tensor[i].unsqueeze(0)).detach().numpy()
    else:
        action = micro_model_up(X_test_tensor[i].unsqueeze(0)).detach().numpy()
    final_actions.append(action.argmax())  # Fix: Store only the most confident action

# Convert actions into trading decisions
def interpret_actions(actions):
    """Convert model outputs into buy/sell/hold signals."""
    mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}  # Adjust mapping if necessary
    return [mapping[a] for a in actions]  # Ensure correct mapping

# Align predictions with data
df = df.iloc[window_size:]
df['Trading_Signal'] = interpret_actions(final_actions)

# Calculate strategy returns
df['Return'] = df['close'].pct_change()
df['Strategy_Return'] = df['Return'] * df['Trading_Signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})

# Set initial capital
total_capital = 10000  # Initial investment in Rs
cumulative_strategy_return = total_capital * (1 + df['Strategy_Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(10, 5))
plt.plot(cumulative_strategy_return, label="Strategy Return", color="blue")
plt.axhline(y=total_capital, color='gray', linestyle='--', label="Initial Capital")
plt.xlabel("Time")
plt.ylabel("Portfolio Value (Rs)")
plt.title("Trading Strategy Performance")
plt.legend()
plt.savefig(f"Strategy_Performance-{stock}.png")
plt.show()
csv_path=f"./Dataset/Evaluation/{stock}_predictions.csv"
df.to_csv(csv_path, index=False)
print("Predictions saved to:", f"{stock}_predictions.csv")
print("Strategy performance plot saved ")
