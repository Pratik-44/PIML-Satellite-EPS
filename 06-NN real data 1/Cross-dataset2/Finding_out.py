'''import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

train_file = "F:/Downloads/PIML_Results/NN real data/schedule_predictions.csv"
test_file  = "F:/Downloads/PIML_Results/NN real data/feb2026rb2r2_updated.txt"

train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values("timestamp")
test_df  = test_df.sort_values("timestamp")

# =========================
# OPTIONAL DOWNSAMPLING
# =========================
'''
'''
STEP = 20
train_df = train_df.iloc[::STEP]
test_df  = test_df.iloc[::STEP]
'''
'''
# =========================
# FUNCTION
# =========================

def plot_single_feature(feature):

    plt.figure(figsize=(14,5))

    plt.plot(train_df["timestamp"], train_df[feature],
             label="Train", color="blue", linewidth=1)

    plt.plot(test_df["timestamp"], test_df[feature],
             label="Test", color="orange", linewidth=1)

    plt.title(f"{feature} comparison (Train vs Test)")
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{feature}_train_vs_test.png", dpi=300)
    plt.show()

# =========================
# CALL SEPARATELY
# =========================

plot_single_feature("voltage")
plot_single_feature("current")
plot_single_feature("heater_power")
plot_single_feature("load_current")
plot_single_feature("total_power")
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# LOAD DATA
# =========================

train_file = "F:/Downloads/PIML_Results/NN real data/schedule_predictions.csv"
test_file  = "F:/Downloads/PIML_Results/NN real data/feb2026rb2r2_updated.txt"

train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values("timestamp")
test_df  = test_df.sort_values("timestamp")

# =========================
# OPTIONAL DOWNSAMPLING
# =========================

STEP = 20   # adjust if needed

train_plot = train_df.iloc[::STEP]
test_plot  = test_df.iloc[::STEP]

# =========================
# FUNCTION (SUBPLOTS)
# =========================

def plot_feature_subplots(feature):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=False
    )

    # =====================
    # TRAIN PLOT
    # =====================
    ax1.plot(
        train_plot["timestamp"],
        train_plot[feature],
        color="blue",
        linewidth=1,
        label="Train"
    )
    ax1.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m-%d %H')
)

    ax1.set_title(f"{feature} - Train Dataset")
    ax1.set_ylabel(feature)
    ax1.legend()
    ax1.grid()

    # =====================
    # TEST PLOT
    # =====================
    ax2.plot(
        test_plot["timestamp"],
        test_plot[feature],
        color="orange",
        linewidth=1,
        label="Test"
    )

    ax2.set_title(f"{feature} - Test Dataset")
    ax2.set_xlabel("Time")
    ax2.set_ylabel(feature)
    ax2.legend()
    ax2.grid()

    # =====================
    # SAVE + SHOW
    # =====================
    plt.tight_layout()
    plt.savefig(f"{feature} train_vs_test.png", dpi=300)
    plt.show()

# =========================
# CALL FOR ALL FEATURES
# =========================

features = [
    "voltage",
    "current",
    "heater_power",
    "load_current",
    "total_power"
]

for feature in features:
    plot_feature_subplots(feature)