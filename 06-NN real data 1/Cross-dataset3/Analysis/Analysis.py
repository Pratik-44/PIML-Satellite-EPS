import pandas as pd
import numpy as np

# Select target
TARGET = "total_power"   # "voltage" OR "total_power"

# Select prediction file
PRED_FILE = "F:/Downloads/PIML_Results/NN real data/Cross-dataset3/Power Prediction/power_LSTM_predictions.csv"

# Test dataset
TEST_FILE = "F:/Downloads/PIML_Results/NN real data/feb2026rb2r2_updated.txt"

# =========================
# 1. LOAD DATA
# =========================

test_df = pd.read_csv(TEST_FILE)
pred_df = pd.read_csv(PRED_FILE)

test_df.columns = test_df.columns.str.strip()
pred_df.columns = pred_df.columns.str.strip()

test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

test_df = test_df.sort_values("timestamp").reset_index(drop=True)
pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)

# =========================
# 2. MERGE
# =========================

df = pd.merge(
    pred_df,
    test_df,
    on="timestamp",
    how="left"
)

print("Merged shape:", df.shape)

# =========================
# 3. FEATURES
# =========================

feature_cols = [
    "current",
    "heater_power",
    "load_current",
    "eclipse_flag",
    "payload_status"
]

# =========================
# 4. ERROR
# =========================

error = np.abs(df["Actual"] - df["Predicted"])

# =========================
# 5. FEATURE vs ERROR CORRELATION
# =========================

print("\n=== FEATURE vs ERROR CORRELATION ===")

for col in feature_cols:
    corr = np.corrcoef(df[col], error)[0,1]
    print(f"{col:15s} : {corr:.4f}")

# =========================
# 6. DISTRIBUTION (MEAN & STD)
# =========================

print("\n=== TEST DATA DISTRIBUTION ===")

for col in feature_cols:
    print(f"\n{col}")
    print(f"Mean: {df[col].mean():.4f}")
    print(f"Std : {df[col].std():.4f}")

# =========================
# 7. RANGE
# =========================

print("\n=== RANGE ===")

for col in feature_cols:
    print(f"\n{col}")
    print(f"{df[col].min():.4f} → {df[col].max():.4f}")

# =========================
# 8. ERROR vs FEATURE (GROUP ANALYSIS)
# =========================

print("\n=== ERROR BY FEATURE BINS ===")

for col in feature_cols:
    try:
        df["bin"] = pd.qcut(df[col], q=5, duplicates='drop')
        grouped = df.groupby("bin", observed=True)[["Actual", "Predicted"]].mean()

        err = np.abs(grouped["Actual"] - grouped["Predicted"]).mean()

        print(f"{col:15s} → Avg Error: {err:.4f}")

    except:
        print(f"{col:15s} → Skipped (constant or categorical)")