import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# MODEL DATA PREPARATION (ML / PIML)
# =====================================================
def prepare_model_df(file_path):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# =====================================================
# SINGLE MODEL: Eclipse Behaviour
# =====================================================
def plot_eclipse_behavior(df, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=True
    )

    # ===== Total Power =====
    ax1.plot(
        df["timestamp"],
        df["total_power"],
        color="blue",
        label="Total Power"
    )

    # ===== Eclipse shading (from SAME model) =====
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

    ax1.set_ylabel("Power (W)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid()

    # ===== Payload =====
    ax2.step(
        df["timestamp"],
        df["payload_status"],
        where="post",
        color="red",
        label="Payload Status"
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
# SINGLE MODEL: Heater
# =====================================================
def plot_heater(df, title):

    plt.figure(figsize=(14,3))

    plt.plot(
        df["timestamp"],
        df["heater_power"],
        color="orange",
        linewidth=1,
        label="Heater Power"
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

# ----- ML -----
ml_df = prepare_model_df(
    "F:/Downloads/PIML_Results/New Data/het_mlout/long_rollout_12h_lstm_ml.csv"
)

# ----- PIML -----
piml_df = prepare_model_df(
    "F:/Downloads/PIML_Results/New Data/het_piml_out/long_rollout_12h_lstm_piml.csv"
)

# =====================================================
# PLOTS (SEPARATE)
# =====================================================

# ===== ML =====
plot_eclipse_behavior(
    ml_df,
    title="ML_Eclipse_Behavior"
)

plot_heater(
    ml_df,
    title="ML_Heater_Power"
)

# ===== PIML =====
plot_eclipse_behavior(
    piml_df,
    title="PIML_Eclipse_Behavior"
)

plot_heater(
    piml_df,
    title="PIML_Heater_Power"
)