import joblib
import pandas as pd

# 1) Load data (same CSV as training)
df = pd.read_csv("Market price of electricity (RCE) 2025-08-14 - 2025-12-01.csv")
df.columns = [c.replace("\n", "") for c in df.columns]
df["OREB[Time unit from - to]"] = df["OREB[Time unit from - to]"].str.replace("a", "", regex=False)
df["clean_time"] = df["OREB[Time unit from - to]"].str.extract(r"(\d{2}:\d{2})")
df = df.dropna(subset=["clean_time"]).reset_index(drop=True)
df["time"] = pd.to_datetime(df["Date"] + " " + df["clean_time"])
df = df[["time", "RCE[zł/MWh]"]]
df.columns = ["time", "price"]

# 2) Apply the same feature engineering (must match training)
df["hour"] = df["time"].dt.hour
df["minute"] = df["time"].dt.minute
df["dayofweek"] = df["time"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["price_lag1"] = df["price"].shift(1)
df["price_lag4"] = df["price"].shift(4)
df["price_lag96"] = df["price"].shift(96)
df["rolling_mean_96"] = df["price"].shift(1).rolling(96).mean()
df["rolling_std_96"] = df["price"].shift(1).rolling(96).std()
df = df.dropna().reset_index(drop=True)

# 3) Load the deployment bundle
bundle = joblib.load("best_model_deploy.joblib")
rf = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

# 4) Take the latest feature row and predict
X_new = df.drop(columns=["price", "time"]).iloc[-1:][feature_cols]
X_new_scaled = scaler.transform(X_new)
pred = rf.predict(X_new_scaled)[0]

print(" Deployed prediction (demo):", pred)