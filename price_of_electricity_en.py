import pandas as pd
import numpy as np


from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ====================================================
# 1. Load the RCE 15-min CSV file
# ====================================================

df = pd.read_csv("Market price of electricity (RCE) 2025-08-14 - 2025-12-01.csv")

# 2. Remove newline characters from column names
df.columns = [c.replace("\n", "") for c in df.columns]

# Column names become:
# 'Date'
# 'OREB[Time unit from - to]'
# 'RCE[zł/MWh]'

# Fix DST time strings like 02a:xx
df["OREB[Time unit from - to]"] = df["OREB[Time unit from - to]"].str.replace("a", "", regex=False)

# Extract the time again
df["clean_time"] = df["OREB[Time unit from - to]"].str.extract(r"(\d{2}:\d{2})")

# Drop remaining bad rows (if any)
df = df.dropna(subset=["clean_time"]).reset_index(drop=True)

# Combine date and time into a full timestamp
df["time"] = pd.to_datetime(df["Date"] + " " + df["clean_time"])

# 7. Keep only time and price columns
df = df[["time", "RCE[zł/MWh]"]]
df.columns = ["time", "price"]

# ========== End of time parsing; continue with feature engineering ==========


# ====================================================
# 2. Feature engineering (for 15-minute electricity prices)
# ====================================================

# Basic time features
df["hour"] = df["time"].dt.hour
df["minute"] = df["time"].dt.minute
df["dayofweek"] = df["time"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# Lag features (very important)
df["price_lag1"] = df["price"].shift(1)    # 15 minutes ago
df["price_lag4"] = df["price"].shift(4)    # 1 hour ago
df["price_lag96"] = df["price"].shift(96)  # 24 hours ago (96×15min)

# Rolling window features
df["rolling_mean_96"] = df["price"].shift(1).rolling(96).mean()  # 24-hour rolling mean
df["rolling_std_96"] = df["price"].shift(1).rolling(96).std()    # 24-hour rolling standard deviation (volatility)

# Drop rows with NaN introduced by lag/rolling features
df = df.dropna().reset_index(drop=True)

print("Rows:", len(df))
print("Time range:", df["time"].min(), "to", df["time"].max())
print(df.head(3))
# ====================================================
# 3. Split train/test sets (time-ordered)
# ====================================================

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train.drop(columns=["price", "time"])
y_train = train["price"]

X_test = test.drop(columns=["price", "time"])
y_test = test["price"]


# ====================================================
# 4. Feature scaling (required for KNN)
# ====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ====================================================
# 5. Train models (traditional regressors)
# ====================================================

models = {
    "Ridge": Ridge(alpha=1.0),
    "SVR (RBF)": SVR(kernel="rbf", C=1, epsilon=0.1, gamma="scale"),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=8),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12)
}

results = {}


best_name = None
best_mae = float("inf")
best_model = None
for name, model in models.items():
    print(f"\n===== Training {name} =====")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 6. Store and print final results
    # =================================================
    results[name] = (mae, rmse)
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    if mae < best_mae:
        best_mae = mae
        best_name = name
        best_model = model
print(f"\n Best model: {best_name} (MAE={best_mae:.3f})")


# ====================================================
# ================== Save the selected best model (deployment) ==================
import joblib
feature_cols = X_train.columns.tolist()
joblib.dump(
    {"model": best_model, "scaler": scaler, "feature_cols": feature_cols},
    "best_model_deploy.joblib"
)
print(f" Saved: best_model_deploy.joblib ({best_name})")


# ================== train Best model  + predict + plot ==================

# Predict with the best model for visualization
y_pred_best = best_model.predict(X_test_scaled)

# Create result table for inspection
result = test.copy()
result["price_pred"] = y_pred_best

print("\nSample predictions (first 5 rows of test):")
print(result[["time", "price", "price_pred"]].head(5))
# Plot last 200 points
plot_df = result.tail(200)

plt.figure(figsize=(12, 4))
plt.plot(plot_df["time"], plot_df["price"], label="Real price")
plt.plot(plot_df["time"], plot_df["price_pred"], label=f"Predicted ({best_name})")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Price (zł/MWh)")
plt.title(f"Best Model ({best_name}) - Real vs Predicted (last 200 points)")
plt.legend()
plt.tight_layout()
plt.show()