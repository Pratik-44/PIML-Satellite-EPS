import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
print("Voltage mean:", df["voltage"].mean())
print("Voltage std :", df["voltage"].std())
print("Voltage min :", df["voltage"].min())
print("Voltage max :", df["voltage"].max())

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time (VERY IMPORTANT)
df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset shape:", df.shape)

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

target_col = "voltage"

X = df[feature_cols]
y = df[target_col]
time = df["timestamp"]

def create_sequences(X, y, timestamps, window_size=10):
    Xs, ys, ts = [], [], []

    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
        ts.append(timestamps[i+window_size])

    return np.array(Xs), np.array(ys), np.array(ts)

# =========================
# 3. TRAIN TEST SPLIT (NO SHUFFLE)
# =========================

X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, time,
    test_size=0.2,
    shuffle=False
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# 4. SCALING
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.array(X_train_scaled)
X_test_scaled = np.array(X_test_scaled)

y_train = np.array(y_train)
y_test = np.array(y_test)
t_train = np.array(t_train)
t_test = np.array(t_test)

window_size = 10

X_train_seq, y_train_seq, t_train_seq = create_sequences(
    X_train_scaled, y_train, t_train, window_size
)

X_test_seq, y_test_seq, t_test_seq = create_sequences(
    X_test_scaled, y_test, t_test, window_size
)

print("LSTM Train shape:", X_train_seq.shape)
print("LSTM Test shape :", X_test_seq.shape)

# =========================
# 5. BUILD LSTM MODEL
# =========================

model = keras.Sequential([
    layers.LSTM(64, input_shape=(window_size, X_train_seq.shape[2])),
    layers.Dense(32, activation="relu"),
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
    X_train_seq,
    y_train_seq,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 7. PREDICT
# =========================

y_pred = model.predict(X_test_seq).flatten()

# =========================
# 8. METRICS
# =========================

mae = mean_absolute_error(y_test_seq, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
r2 = r2_score(y_test_seq, y_pred)

print("\n=== Voltage LSTM Results ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# =========================
# 9. SAVE PREDICTIONS CSV
# =========================

results_df = pd.DataFrame({
    "timestamp": t_test_seq,
    "Actual": y_test_seq,
    "Predicted": y_pred
})

results_df.to_csv("voltage_LSTM_predictions.csv", index=False)
print("Saved: voltage_LSTM_predictions.csv")

