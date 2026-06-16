import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =========================
# 1. LOAD DATA
# =========================
file_path = "your_dataset.txt"   # ← change this
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# =========================
# 2. TIMESTAMP PROCESSING
# =========================
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Basic time features
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["second"] = df["timestamp"].dt.second

# Cyclic encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# =========================
# 3. HIGH-VALUE ENGINEERED FEATURES
# =========================

# Eclipse transition detector
df["eclipse_change"] = df["eclipse_flag"].diff().fillna(0).abs()

# Rolling load behavior
df["load_current_roll3"] = (
    df["load_current"]
    .rolling(window=3)
    .mean()
    .bfill()
)

# =========================
# 4. DROP UNUSED COLUMNS
# =========================
df = df.drop(columns=["timestamp", "hour", "minute", "second"])

# =========================
# 5. FEATURE SELECTION (POWER SAFE)
# =========================
features = [
    "heater_power",
    "load_current",
    "eclipse_flag",
    "payload_status",
    "aoce_mode",
    "hour_sin",
    "hour_cos",
    "eclipse_change",
    "load_current_roll3"
]

# 🔥 Target changed to power
target = "total_power"

X = df[features]
y = df[target]

# =========================
# 6. TIME-BASED TRAIN TEST SPLIT
# =========================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# =========================
# 7. MODELS
# =========================
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=120,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
}

# =========================
# 8. TRAIN + EVALUATE
# =========================
results = []
predictions_store = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    predictions_store[name] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

# =========================
# 9. RESULTS TABLE
# =========================
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
).sort_values("R2", ascending=False)

print("\n===== MODEL COMPARISON =====")
print(results_df)

# =========================
# 10. BEST MODEL IDENTIFICATION
# =========================
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest performing model: {best_model_name}")