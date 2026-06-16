import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# 1. LOAD DATA
# =========================

train_file = "F:/Downloads/PIML_Results/NN real data/schedule_predictions.csv"
test_file  = "F:/Downloads/PIML_Results/NN real data/feb2026rb2r2_updated.txt"

train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values("timestamp").reset_index(drop=True)
test_df  = test_df.sort_values("timestamp").reset_index(drop=True)

# =========================
# 2. FEATURES
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

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]
t_test = test_df["timestamp"]

# =========================
# 3. SCALING
# =========================

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled  = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1,1)).flatten()
y_test_scaled  = y_scaler.transform(y_test.values.reshape(-1,1)).flatten()

# =========================
# 4. SEQUENCE CREATION
# =========================

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 30   # Increased

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
X_test_seq, y_test_seq   = create_sequences(X_test_scaled, y_test_scaled, SEQ_LEN)

# =========================
# 5. POSITIONAL ENCODING
# =========================

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_emb = self.add_weight(
            shape=(sequence_length, d_model),
            initializer="random_normal",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_emb

# =========================
# 6. TRANSFORMER BLOCK
# =========================

def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)

    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(x.shape[-1])(ffn)

    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

# =========================
# 7. BUILD MODEL
# =========================

inputs = keras.Input(shape=(SEQ_LEN, X_train_seq.shape[2]))

x = PositionalEncoding(SEQ_LEN, X_train_seq.shape[2])(inputs)

# STACKED TRANSFORMER
x = transformer_block(x, 32, 4, 128)
x = transformer_block(x, 32, 4, 128)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)

outputs = layers.Dense(1)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================
# 8. TRAIN
# =========================

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=60,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 9. PREDICT
# =========================

y_pred_scaled = model.predict(X_test_seq).flatten()

y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_test_actual = y_scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()

# =========================
# 10. METRICS
# =========================

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("\n=== Voltage Transformer Results ===")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# =========================
# 11. SAVE
# =========================

results_df = pd.DataFrame({
    "timestamp": t_test.values[SEQ_LEN:],
    "Actual": y_test_actual,
    "Predicted": y_pred
})

results_df.to_csv("voltage_transformer_predictions.csv", index=False)