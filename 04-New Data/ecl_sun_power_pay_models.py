import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# ORIGINAL DATA PREPARATION
# =====================================================
def prepare_original_df(file_path):

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Parse timestamp
    df["Time"] = pd.to_datetime(
        df["Time"],
        format="%d-%m-%Y-%H:%M:%S:%f"
    )

    df = df.sort_values("Time").reset_index(drop=True)

    # Eclipse flag
    df["eclipse_flag"] = (
        df["FDD08633:S/C_SUNLIT_ECL"] < 0.5
    ).astype(int)

    # Standard naming
    df["total_power"] = df["PWRS0152:SC_LOAD_PWR_W"]
    df["payload_status"] = df["PLDHC041:PLC_M_EPC_STS"]
    df["heater_power"] = df["THRS0001:HTR_LOAD_ON_BUS-A"]
    df["timestamp"] = df["Time"]

    return df


# =====================================================
# MODEL DATA PREPARATION (ML / PIML)
# =====================================================
def prepare_model_df(file_path):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# =====================================================
# COMPARISON: Eclipse Behaviour
# =====================================================
def plot_eclipse_comparison(orig_df, model_df, model_name, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=True
    )

    # ===== Total Power =====
    ax1.plot(
        orig_df["timestamp"],
        orig_df["total_power"],
        color="blue",
        label="Total Power (Original)"
    )

    ax1.plot(
        model_df["timestamp"],
        model_df["total_power"],
        color="blue",
        linestyle="--",
        label=f"Total Power ({model_name})"
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

    ax1.set_ylabel("Power (W)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid()

    # ===== Payload =====
    ax2.step(
        orig_df["timestamp"],
        orig_df["payload_status"],
        where="post",
        color="red",
        label="Payload (Original)"
    )

    ax2.step(
        model_df["timestamp"],
        model_df["payload_status"],
        where="post",
        color="red",
        linestyle="--",
        label=f"Payload ({model_name})"
    )

    ax2.set_ylabel("Payload")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


# =====================================================
# COMPARISON: Heater
# =====================================================
def plot_heater_comparison(orig_df, model_df, model_name, title):

    plt.figure(figsize=(14,3))

    plt.plot(
        orig_df["timestamp"],
        orig_df["heater_power"],
        color="orange",
        linewidth=1,
        label="Heater (Original)"
    )

    plt.plot(
        model_df["timestamp"],
        model_df["heater_power"],
        color="orange",
        linestyle="--",
        linewidth=1,
        label=f"Heater ({model_name})"
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Heater Power (W)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


# =====================================================
# ====================== MAIN =========================
# =====================================================

# ----- Original -----
orig_df = prepare_original_df("samedata_r2.txt")

# ----- ML -----
ml_df = prepare_model_df(
    "F:/Downloads/PIML_Results/New Data/het_mlout/long_rollout_12h_lstm_ml.csv"
)

# ----- PIML -----
piml_df = prepare_model_df(
    "F:/Downloads/PIML_Results/New Data/het_piml_out/long_rollout_12h_lstm_piml.csv"
)

# =====================================================
# PLOTS
# =====================================================

# Original vs ML
plot_eclipse_comparison(
    orig_df, ml_df,
    model_name="ML",
    title="Original_vs_ML_Eclipse_Behavior"
)

plot_heater_comparison(
    orig_df, ml_df,
    model_name="ML",
    title="Original_vs_ML_Heater"
)

# Original vs PIML
plot_eclipse_comparison(
    orig_df, piml_df,
    model_name="PIML",
    title="Original_vs_PIML_Eclipse_Behavior"
)

plot_heater_comparison(
    orig_df, piml_df,
    model_name="PIML",
    title="Original_vs_PIML_Heater"
)