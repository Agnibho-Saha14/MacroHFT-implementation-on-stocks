import numpy as np
import pandas as pd
df=pd.read_csv("./Dataset/Training/preprocessed_data.csv")
# Market Trend Label (Bullish = 1, Bearish = 0)
df["Trend"] = np.where(df["close"] > df["SMA_10"], 1, 0)

# Market Volatility Label (Stable = 0, Medium = 1, High = 2)
vol_quantiles = df["ATR"].quantile([0.33, 0.66])
df["Volatility_Label"] = np.where(
    df["ATR"] <= vol_quantiles[0.33], 0,  
    np.where(df["ATR"] <= vol_quantiles[0.66], 1, 2)
)

df.to_csv("./Dataset/Training/ETHUSDT_labeled.csv", index=False)
print("Labeling complete")
