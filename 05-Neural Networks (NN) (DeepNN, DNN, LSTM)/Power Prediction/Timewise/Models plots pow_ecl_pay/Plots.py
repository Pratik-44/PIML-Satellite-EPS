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

# -------------------------------------------------
# OPTIONAL (COMMENTED) — eclipse inversion logic
# -------------------------------------------------
orig_df["eclipse_flag"] = (
    orig_df["eclipse_flag"] < 0.5
).astype(int)

print("Original loaded:", orig_df.shape)


# =========================
# 2. FUNCTION TO MERGE PREDICTIONS
# =========================

def load_and_merge(pred_file):

    pred_df = pd.read_csv(pred_file)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

    # 🔴 CRITICAL STEP — timestamp alignment
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
    # Heater Power and Payload plots
    # =====================
    # LEFT → Heater power
    line1, = ax2.plot(
        df["timestamp"],
        df["heater_power"],
        color="green",
        linewidth=1,
        label="Heater Power"
    )
    ax2.set_ylabel("Heater Power (W)")
    ax2.grid()

    # RIGHT → Payload
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
# 4. LOAD EACH MODEL
# =========================

dnn_df = load_and_merge("F:/Downloads/PIML_Results/NN/Power Prediction/60-40/power_DNN_predictions.csv")
deepnn_df = load_and_merge("F:/Downloads/PIML_Results/NN/Power Prediction/60-40/power_DeepNN_predictions.csv")
lstm_df = load_and_merge("F:/Downloads/PIML_Results/NN/Power Prediction/60-40/power_LSTM_predictions.csv")

print("DNN merged:", dnn_df.shape)
print("DeepNN merged:", deepnn_df.shape)
print("LSTM merged:", lstm_df.shape)

# =========================
# 5. PLOT — THREE FIGURES
# =========================

plot_power_model(dnn_df, "Power_DNN_vs_Original_inverse_ecl_flag")
plot_power_model(deepnn_df, "Power_DeepNN_vs_Original_inverse_ecl_flag")
plot_power_model(lstm_df, "Power_LSTM_vs_Original_inverse_ecl_flag")