import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# ORIGINAL DATA PREPARATION
# =====================================================
def prepare_original_df(file_path):

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    df["Time"] = pd.to_datetime(
        df["Time"],
        format="%d-%m-%Y-%H:%M:%S:%f"
    )

    df = df.sort_values("Time").reset_index(drop=True)

    # Raw: 1 = sunlit, 0 = eclipse
    # Needed: 1 = eclipse, 0 = sunlight
    df["eclipse_flag"] = (
        df["FDD08633:S/C_SUNLIT_ECL"] < 0.5
    ).astype(int)

    df["timestamp"] = df["Time"]

    return df


# =====================================================
# MODEL DATA PREPARATION
# =====================================================
def prepare_model_df(file_path):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# =====================================================
# COMPARISON PLOT
# =====================================================
def plot_eclipse_comparison(orig_df, model_df, model_name, title):

    plt.figure(figsize=(14,3))

    # Original
    plt.plot(
        orig_df["timestamp"],
        orig_df["eclipse_flag"],
        linewidth=1,
        label="Eclipse Flag (Original)"
    )

    # Model
    plt.plot(
        model_df["timestamp"],
        model_df["eclipse_flag"],
        linewidth=1,
        linestyle="--",
        label=f"Eclipse Flag ({model_name})"
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

plot_eclipse_comparison(
    orig_df,
    ml_df,
    model_name="ML",
    title="Original_vs_ML_Eclipse_Flag"
)

plot_eclipse_comparison(
    orig_df,
    piml_df,
    model_name="PIML",
    title="Original_vs_PIML_Eclipse_Flag"
)