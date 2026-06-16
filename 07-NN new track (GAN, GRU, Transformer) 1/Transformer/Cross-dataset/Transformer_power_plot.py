import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. LOAD ORIGINAL DATA
# =========================

orig_file = "F:/Downloads/PIML_Results/NN/samedata_r2_updated.txt"
orig_df = pd.read_csv(orig_file)
orig_df.columns = orig_df.columns.str.strip()

orig_df["timestamp"] = pd.to_datetime(orig_df["timestamp"])
orig_df = orig_df.sort_values("timestamp").reset_index(drop=True)

# Eclipse inversion
orig_df["eclipse_flag"] = (orig_df["eclipse_flag"] < 0.5).astype(int)

print("Original loaded:", orig_df.shape)

# =========================
# 2. MERGE FUNCTION
# =========================

def load_and_merge(pred_file):

    pred_df = pd.read_csv(pred_file)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

    merged = pd.merge(
        pred_df,
        orig_df[["timestamp", "payload_status", "eclipse_flag", "heater_power"]],
        on="timestamp",
        how="left"
    )

    return merged

# =========================
# 3. PLOT FUNCTION
# =========================

def plot_power_model(df, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=True
    )

    # =====================
    # Power plot
    # =====================
    ax1.plot(
        df["timestamp"],
        df["Actual"],
        label="Original Power",
        color="blue",
        linewidth=1
    )

    ax1.plot(
        df["timestamp"],
        df["Predicted"],
        label="Predicted Power",
        color="orange",
        linewidth=1
    )

    # =====================
    # Eclipse shading
    # =====================
    eclipse = df["eclipse_flag"].values
    time = df["timestamp"].values

    in_eclipse = False
    start_time = None

    for i in range(len(df)):

        if eclipse[i] == 1 and not in_eclipse:
            in_eclipse = True
            start_time = time[i]

        elif eclipse[i] == 0 and in_eclipse:
            ax1.axvspan(
                start_time, time[i],
                color="gray", alpha=0.25
            )
            in_eclipse = False

    if in_eclipse:
        ax1.axvspan(
            start_time, time[-1],
            color="gray", alpha=0.25
        )

    ax1.set_ylabel("Power (W)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid()

    # =====================
    # Heater Power + Payload
    # =====================
    line1, = ax2.plot(
        df["timestamp"],
        df["heater_power"],
        color="green",
        linewidth=1,
        label="Heater Power"
    )

    ax2.set_ylabel("Heater Power (W)")
    ax2.grid()

    ax2_right = ax2.twinx()

    line2, = ax2_right.step(
        df["timestamp"],
        df["payload_status"],
        where="post",
        color="red",
        label="Payload Status"
    )

    ax2_right.set_ylabel("Payload Status")
    ax2_right.set_ylim(-0.05, 1.05)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()

# =========================
# 4. LOAD TRANSFORMER FILE
# =========================

trans_df = load_and_merge("F:/Downloads/PIML_Results/NN new track/Transformer/Cross-dataset/power_transformer_predictions.csv")

print("Transformer merged:", trans_df.shape)

# =========================
# 5. PLOT
# =========================

plot_power_model(trans_df, "Power_Transformer_vs_Original(orig_mod_timecorr vs samedata_r2_updated)")