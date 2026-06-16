import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# LOAD DATA
# =========================

train_file = "F:/Downloads/PIML_Results/NN real data/trainset_synth.csv"
test_file  = "F:/Downloads/PIML_Results/NN real data/feb2026rb2r2_updated.txt"

train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"])

train_df = train_df.sort_values("timestamp").reset_index(drop=True)
test_df  = test_df.sort_values("timestamp").reset_index(drop=True)

# =========================
# MATCH TEST DURATION TO TRAIN
# =========================

train_duration = train_df["timestamp"].max() - train_df["timestamp"].min()

test_start = test_df["timestamp"].min()
test_end = test_start + train_duration

test_df_equal = test_df[
    (test_df["timestamp"] >= test_start) &
    (test_df["timestamp"] <= test_end)
].reset_index(drop=True)

print("Train duration:", train_duration)
print("Test sliced shape:", test_df_equal.shape)

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
        train_df["timestamp"],
        train_df[feature],
        color="blue",
        linewidth=1,
        label="Train"
    )

    # 🔶 Eclipse shading (TRAIN)
    eclipse = train_df["eclipse_flag"].values
    time = train_df["timestamp"].values

    in_eclipse = False
    start_time = None

    for i in range(len(train_df)):
        if eclipse[i] == 1 and not in_eclipse:
            in_eclipse = True
            start_time = time[i]
        elif eclipse[i] == 0 and in_eclipse:
            ax1.axvspan(start_time, time[i], color="gray", alpha=0.25)
            in_eclipse = False

    if in_eclipse:
        ax1.axvspan(start_time, time[-1], color="gray", alpha=0.25)

    # Time formatting (train)
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d %H')
    )

    ax1.set_title(f"{feature} - Train Dataset (trainset_synth)")
    ax1.set_ylabel(feature)
    ax1.legend()
    ax1.grid()

    # =====================
    # TEST PLOT (SLICED)
    # =====================
    ax2.plot(
        test_df_equal["timestamp"],
        test_df_equal[feature],
        color="orange",
        linewidth=1,
        label="Test"
    )

    # 🔶 Eclipse shading (TEST SLICED)
    eclipse = test_df_equal["eclipse_flag"].values
    time = test_df_equal["timestamp"].values

    in_eclipse = False
    start_time = None

    for i in range(len(test_df_equal)):
        if eclipse[i] == 1 and not in_eclipse:
            in_eclipse = True
            start_time = time[i]
        elif eclipse[i] == 0 and in_eclipse:
            ax2.axvspan(start_time, time[i], color="gray", alpha=0.25)
            in_eclipse = False

    if in_eclipse:
        ax2.axvspan(start_time, time[-1], color="gray", alpha=0.25)

    # Time formatting (test)
    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d %H')
    )

    ax2.set_title(f"{feature} - Test Dataset (feb2026rb2r2_updated) (equal duration)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel(feature)
    ax2.legend()
    ax2.grid()

    # =====================
    # SAVE + SHOW
    # =====================
    plt.tight_layout()
    plt.savefig(f"{feature}_train_vs_test_equal_duration.png", dpi=300)
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