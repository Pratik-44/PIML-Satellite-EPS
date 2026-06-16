import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# ORIGINAL DATA PREPARATION
# ===============================
def prepare_original_df(df):

    df = df.copy()
    df.columns = df.columns.str.strip()

    df["Time"] = pd.to_datetime(
        df["Time"],
        format="%d-%m-%Y-%H:%M:%S:%f"
    )

    df = df.sort_values("Time").reset_index(drop=True)

    df["eclipse_flag"] = (
        df["FDD08633:S/C_SUNLIT_ECL"] < 0.5
    ).astype(int)

    df["timestamp"] = df["Time"]
    df["battery_current"] = df["PWRS0151:BAT_DISCHG_CUR_AMP"]
    df["battery_voltage"] = df["PWRS0095:BAT_VOL_FINE_SEL_RT"]
    df["payload_status"] = df["PLDHC041:PLC_M_EPC_STS"]
    df["heater_power"] = df["THRS0001:HTR_LOAD_ON_BUS-A"]

    return df


# ===============================
# MODEL DATA PREPARATION
# ===============================
def prepare_model_df(df):

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["battery_current"] = df["current"]
    df["battery_voltage"] = df["voltage"]

    return df


# ===============================
# COMPARISON PLOT FUNCTION
# ===============================
def plot_comparison(orig_df, model_df, model_name, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True
    )

    # ===== TOP: Current & Voltage =====
    ax1.plot(
        orig_df["timestamp"],
        orig_df["battery_current"],
        color="blue",
        linewidth=1,
        label="Battery Current (Original)"
    )

    ax1.plot(
        model_df["timestamp"],
        model_df["battery_current"],
        color="blue",
        linestyle="--",
        linewidth=1,
        label=f"Battery Current ({model_name})"
    )

    ax1.plot(
        orig_df["timestamp"],
        orig_df["battery_voltage"],
        color="green",
        linewidth=1,
        label="Battery Voltage (Original)"
    )

    ax1.plot(
        model_df["timestamp"],
        model_df["battery_voltage"],
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Battery Voltage ({model_name})"
    )

    # ===== Eclipse shading (from ORIGINAL) =====
    eclipse = orig_df["eclipse_flag"].values
    time = orig_df["timestamp"].values

    in_eclipse = False
    start_time = None

    for i in range(len(orig_df)):

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

    # ===== BOTTOM: Heater =====
    line_heater_orig, = ax2.plot(
        orig_df["timestamp"],
        orig_df["heater_power"],
        color="orange",
        linewidth=1,
        label="Heater Power (Original)"
    )

    line_heater_model, = ax2.plot(
        model_df["timestamp"],
        model_df["heater_power"],
        color="orange",
        linestyle="--",
        linewidth=1,
        label=f"Heater Power ({model_name})"
    )

    ax2.set_ylabel("Heater Power (W)")
    ax2.grid()

    # ===== Payload (right axis) =====
    ax2r = ax2.twinx()

    line_payload_orig, = ax2r.step(
        orig_df["timestamp"],
        orig_df["payload_status"],
        where="post",
        color="red",
        label="Payload Status (Original)"
    )

    line_payload_model, = ax2r.step(
        model_df["timestamp"],
        model_df["payload_status"],
        where="post",
        color="red",
        linestyle="--",
        label=f"Payload Status ({model_name})"
    )

    ax2r.set_ylabel("Payload Status (0/1)")
    ax2r.set_ylim(-0.1, 1.1)

    # combined legend
    lines = [
        line_heater_orig,
        line_heater_model,
        line_payload_orig,
        line_payload_model,
    ]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


# main

# ----- Load original -----
orig_df = pd.read_csv("samedata_r2.txt")
orig_df = prepare_original_df(orig_df)

# ----- Load ML -----
ml_df = pd.read_csv("F:/Downloads/PIML_Results/New Data/het_mlout/long_rollout_12h_lstm_ml.csv")
ml_df = prepare_model_df(ml_df)

# ----- Load PIML -----
piml_df = pd.read_csv("F:/Downloads/PIML_Results/New Data/het_piml_out/long_rollout_12h_lstm_piml.csv")
piml_df = prepare_model_df(piml_df)

# ===============================
# PLOTS
# ===============================

plot_comparison(
    orig_df,
    ml_df,
    model_name="ML",
    title="Original_vs_ML_EPS_Analysis"
)

plot_comparison(
    orig_df,
    piml_df,
    model_name="PIML",
    title="Original_vs_PIML_EPS_Analysis"
)