import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# MODEL DATA PREPARATION
# ===============================
def prepare_model_df(df):

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # map to plotting names (to match your old logic)
    df["battery_current"] = df["current"]
    df["battery_voltage"] = df["voltage"]

    return df


# ===============================
# SINGLE DATASET PLOT
# ===============================
def plot_model_behavior(df, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True
    )

    # ===== Battery Current & Voltage =====
    ax1.plot(
        df["timestamp"],
        df["battery_current"],
        label="Battery Current (A)",
        color="blue",
        linewidth=1
    )

    ax1.plot(
        df["timestamp"],
        df["battery_voltage"],
        label="Battery Voltage (V)",
        color="green",
        linewidth=1
    )

    # ===== Eclipse shading =====
    eclipse = df["eclipse_flag"].values
    time = df["timestamp"].values

    in_eclipse = False
    start_time = None

    for i in range(len(df)):

        if eclipse[i] == 1 and not in_eclipse:
            in_eclipse = True
            start_time = time[i]

        elif eclipse[i] == 0 and in_eclipse:
            ax1.axvspan(start_time, time[i], color="gray", alpha=0.25)
            in_eclipse = False

    if in_eclipse:
        ax1.axvspan(start_time, time[-1], color="gray", alpha=0.25)

    ax1.set_ylabel("Battery Current / Voltage")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.grid()

    # ===== Heater =====
    line_heater, = ax2.plot(
        df["timestamp"],
        df["heater_power"],
        color="orange",
        linewidth=1,
        label="Heater Power (W)"
    )

    ax2.set_ylabel("Heater Power (W)")
    ax2.grid()

    # ===== Payload (right axis) =====
    ax2r = ax2.twinx()

    line_payload, = ax2r.step(
        df["timestamp"],
        df["payload_status"],
        where="post",
        color="red",
        label="Payload Status"
    )

    ax2r.set_ylabel("Payload Status (0/1)")
    ax2r.set_ylim(-0.1, 1.1)

    # combined legend
    lines = [line_heater, line_payload]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


# =====================================================
# ====================== MAIN =========================
# =====================================================

# ----- Load ML -----
ml_df = pd.read_csv(
    "F:/Downloads/PIML_Results/New Data/het_mlout/long_rollout_12h_lstm_ml.csv"
)
ml_df = prepare_model_df(ml_df)

# ----- Load PIML -----
piml_df = pd.read_csv(
    "F:/Downloads/PIML_Results/New Data/het_piml_out/long_rollout_12h_lstm_piml.csv"
)
piml_df = prepare_model_df(piml_df)

# ===============================
# PLOTS (SEPARATE)
# ===============================

plot_model_behavior(
    ml_df,
    title="ML_EPS_Analysis"
)

plot_model_behavior(
    piml_df,
    title="PIML_EPS_Analysis"
)