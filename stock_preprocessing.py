import numpy as np
import pandas as pd
import pandas_ta as ta  

# Load dataset
stock=input("Enter stock ticker to preprocess:")
file_path = f"./Dataset/Evaluation/{stock}.NS_data.csv"
df = pd.read_csv(file_path)

# Convert timestamp to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Set timestamp as index
df.set_index("Timestamp", inplace=True)

# Compute technical indicators
df["SMA_10"] = df.ta.sma(length=10)
df["EMA_10"] = df.ta.ema(length=10)
df["RSI_14"] = df.ta.rsi(length=14)
df["MACD"] = df.ta.macd()["MACD_12_26_9"]

# Compute price differences
df["High-Low"] = df["High"] - df["Low"]
df["High-Close"] = df["High"] - df["Close"].shift(1)
df["Low-Close"] = df["Low"] - df["Close"].shift(1)

# True Range & ATR
df["True Range"] = df.ta.true_range()
df["ATR"] = df.ta.atr(length=14)

# Bollinger Bands
bbands = df.ta.bbands(length=20, std=2)
df["STD_20"] = df["Close"].rolling(window=20).std()
df["Upper_Band"] = bbands["BBU_20_2.0"]
df["Lower_Band"] = bbands["BBL_20_2.0"]

# On-Balance Volume (OBV) & VWAP
df["OBV"] = df.ta.obv()
df["VWAP"] = df.ta.vwap()

# 🔴 Handle Missing Values 🔴
df.fillna(method="ffill", inplace=True)  # Forward fill missing values
df.fillna(method="bfill", inplace=True)  # Backward fill remaining missing values

# Market Trend Label (Bullish = 1, Bearish = 0)
df["Trend"] = np.where(df["Close"] > df["SMA_10"], 1, 0)

# Market Volatility Label (Stable = 0, Medium = 1, High = 2)
vol_quantiles = df["ATR"].quantile([0.33, 0.66])
df["Volatility_Label"] = np.where(
    df["ATR"] <= vol_quantiles[0.33], 0,
    np.where(df["ATR"] <= vol_quantiles[0.66], 1, 2)
)

# Rename columns after processing
df.rename(columns={
    "Timestamp": "timestamp", "Volume": "volume", "Open": "open", "High": "high", "Low": "low", "Close": "close", 
    "SMA_10": "SMA_10", "EMA_10": "EMA_10", "RSI_14": "RSI_14", "MACD": "MACD", "High-Low": "High-Low", 
    "High-Close": "High-Close", "Low-Close": "Low-Close", "True Range": "True Range", "ATR": "ATR", "STD_20": "STD_20", 
    "Upper_Band": "Upper_Band", "Lower_Band": "Lower_Band", "OBV": "OBV", "VWAP": "VWAP", "Trend": "Trend", "Volatility_Label": "Volatility_Label"
}, inplace=True)

# Save processed dataset
processed_file = f"./Dataset/Evaluation/labeled_{stock}.NS.csv"
df.to_csv(processed_file)
print(f"✅ Processed dataset saved as {processed_file}")
