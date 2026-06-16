import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# 1. LOAD DATA (same)
# =========================

train_file = "F:/Downloads/PIML_Results/NN real data/trainset_synth.csv"
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

target_col = "total_power"

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
# 4. SEQUENCE
# =========================

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 30

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
X_test_seq, y_test_seq   = create_sequences(X_test_scaled, y_test_scaled, SEQ_LEN)

# =========================
# 5. GENERATOR
# =========================

def build_generator():
    inputs = keras.Input(shape=(SEQ_LEN, X_train_seq.shape[2]))
    
    x = layers.LSTM(64)(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    
    outputs = layers.Dense(1)(x)
    
    return keras.Model(inputs, outputs)

# =========================
# 6. DISCRIMINATOR
# =========================

def build_discriminator():
    inputs = keras.Input(shape=(1,))
    
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return keras.Model(inputs, outputs)

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0005),
    loss="binary_crossentropy"
)

# =========================
# 7. GAN
# =========================

discriminator.trainable = False

gan_input = keras.Input(shape=(SEQ_LEN, X_train_seq.shape[2]))
fake_output = generator(gan_input)
gan_output = discriminator(fake_output)

gan = keras.Model(gan_input, gan_output)

gan.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="binary_crossentropy"
)

# =========================
# 8. TRAIN
# =========================

EPOCHS = 30
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    idx = np.random.randint(0, X_train_seq.shape[0], BATCH_SIZE)
    
    real_y = y_train_seq[idx]
    fake_y = generator.predict(X_train_seq[idx], verbose=0)
    
    d_loss_real = discriminator.train_on_batch(real_y, np.ones((BATCH_SIZE,1)))
    d_loss_fake = discriminator.train_on_batch(fake_y, np.zeros((BATCH_SIZE,1)))
    
    g_loss = gan.train_on_batch(X_train_seq[idx], np.ones((BATCH_SIZE,1)))
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss_real + d_loss_fake:.4f} | G Loss: {g_loss:.4f}")

# =========================
# 9. PREDICT
# =========================

y_pred_scaled = generator.predict(X_test_seq).flatten()

y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_test_actual = y_scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()

# =========================
# 10. METRICS
# =========================

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("\n=== Power GAN Results ===")
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

results_df.to_csv("power_gan_predictions.csv", index=False)