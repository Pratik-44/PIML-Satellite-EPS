import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# 1. LOAD DATA
# =========================

file_path = "F:/Downloads/PIML_Results/NN/samedata_r2_updated.txt"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
'''
print("Power mean:", df["total_power"].mean())
print("Power std :", df["total_power"].std())
print("Power min :", df["total_power"].min())
print("Power max :", df["total_power"].max())
'''

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time (VERY IMPORTANT)
df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset shape:", df.shape)

#print("Payload status 1 min timestamp:", df[df["payload_status"] == 1]["timestamp"].min())
#print("Payload status 1 max timestamp:", df[df["payload_status"] == 1]["timestamp"].max())

split_time = pd.Timestamp("2024-06-01 11:40:00")

train_df = df[df["timestamp"] < split_time]
test_df  = df[df["timestamp"] >= split_time]

print(len(train_df), len(test_df))
print(train_df["payload_status"].value_counts())
print(test_df["payload_status"].value_counts())

# =========================
# 2. FEATURE SELECTION
# =========================

feature_cols = [
    "current",
    "heater_power",
    "load_current",
    "eclipse_flag",
    "payload_status",
    "aoce_mode"
]

target_col = "total_power"

# Build from split data (IMPORTANT)
X_train = train_df[feature_cols]
y_train = train_df[target_col]
t_train = train_df["timestamp"]

X_test = test_df[feature_cols]
y_test = test_df[target_col]
t_test = test_df["timestamp"]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# 4. SCALING
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. BUILD DNN MODEL
# =========================

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

#early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =========================
# 6. TRAIN
# =========================

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# =========================
# 7. PREDICT
# =========================

y_pred = model.predict(X_test_scaled).flatten()

# =========================
# 8. METRICS
# =========================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Power DNN Results ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# =========================
# 9. SAVE PREDICTIONS CSV
# =========================

results_df = pd.DataFrame({
    "timestamp": t_test.values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

results_df.to_csv("power_DNN_predictions.csv", index=False)

print("Saved: power_DNN_predictions.csv")

