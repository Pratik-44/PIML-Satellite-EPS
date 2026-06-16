import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# MODEL PREPARATION
# ===============================
def prepare_model_df(file_path):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ===============================
# ECLIPSE PLOT FUNCTION
# ===============================
def plot_eclipse_flag(df, title):

    plt.figure(figsize=(14,3))

    plt.plot(
        df["timestamp"],
        df["eclipse_flag"],
        linewidth=1,
        label="Eclipse Flag"
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Eclipse (1) / Sunlight (0)")
    plt.ylim(-0.1, 1.1)
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

# ===============================
# PLOTS
# ===============================

plot_eclipse_flag(
    ml_df,
    title="ML_Eclipse_Flag_vs_Time"
)

plot_eclipse_flag(
    piml_df,
    title="PIML_Eclipse_Flag_vs_Time"
)