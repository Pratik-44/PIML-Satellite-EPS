import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. LOAD DATA

train_file = "F:/Downloads/PIML_Results/NN real data/dec2025rb2r2_updated.txt"
test_file  = "F:/Downloads/PIML_Results/NN real data/feb2026rb2r2_updated.txt"

train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values("timestamp").reset_index(drop=True)
test_df  = test_df.sort_values("timestamp").reset_index(drop=True)

# 2. FEATURE SELECTION

feature_cols = [
    "current",
    "heater_power",
    "load_current",
    "eclipse_flag",
    "payload_status"
]

target_col = "voltage"

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]
t_test = test_df["timestamp"]

# 3. SCALING (INPUT + TARGET)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled  = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1,1)).flatten()
y_test_scaled  = y_scaler.transform(y_test.values.reshape(-1,1)).flatten()

# 4. CREATE SEQUENCES

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 20

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
X_test_seq, y_test_seq   = create_sequences(X_test_scaled, y_test_scaled, SEQ_LEN)

# 5. BUILD GRU MODEL

model = keras.Sequential([
    layers.GRU(64, return_sequences=True, input_shape=(SEQ_LEN, X_train_seq.shape[2])),
    layers.Dropout(0.2),
    layers.GRU(32),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

model.summary()

# 6. TRAIN

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop, lr_scheduler]
)

# 7. PREDICT

y_pred_scaled = model.predict(X_test_seq).flatten()

# Inverse transform
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_test_actual = y_scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()

# 8. METRICS

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

# New metrics
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
max_error = np.max(np.abs(y_test_actual - y_pred))
error_std = np.std(y_test_actual - y_pred)

from sklearn.metrics import median_absolute_error
medae = median_absolute_error(y_test_actual, y_pred)

print("\n Voltage GRU Results: ")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
print("MAPE:", mape)
print("Max Error:", max_error)
print("Median AE:", medae)
print("Error Std:", error_std)

# 9. SAVE RESULTS

results_df = pd.DataFrame({
    "timestamp": t_test.values[SEQ_LEN:],
    "Actual": y_test_actual,
    "Predicted": y_pred
})

results_df.to_csv("voltage_GRU_predictions.csv", index=False)

print("Saved: voltage_GRU_predictions.csv")