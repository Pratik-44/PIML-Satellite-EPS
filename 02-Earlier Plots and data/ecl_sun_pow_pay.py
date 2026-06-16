import pandas as pd
import matplotlib.pyplot as plt

piml_path = "het_piml_out/long_rollout_12h_lstm_piml.csv"
ml_path   = "het_mlout/long_rollout_12h_lstm_ml.csv"

# Load CSVs
piml = pd.read_csv(piml_path, parse_dates=["timestamp"])
ml   = pd.read_csv(ml_path, parse_dates=["timestamp"])

'''
# Downsampling (optional)
step = max(len(piml) // 5000, 1)
piml = piml.iloc[::step]
ml   = ml.iloc[::step]
'''

def plot_eclipse_behavior(df, title):

    # Create stacked subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=True
    )

    # -------------------------
    # TOP: Total Power
    # -------------------------
    ax1.plot(df["timestamp"], df["total_power"],
             label="Total Power", color="blue")

    # Shade eclipse regions
    for i in range(len(df)-1):
        if df["eclipse_flag"].iloc[i] == 1:
            ax1.axvspan(df["timestamp"].iloc[i],
                        df["timestamp"].iloc[i+1],
                        color="gray", alpha=0.3)

    ax1.set_ylabel("Power (W)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid()

    # -------------------------
    # BOTTOM: Payload Status
    # -------------------------
    ax2.step(df["timestamp"], df["payload_status"],
             where="post", color="red",
             label="Payload Status")

    ax2.set_ylabel("Payload")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()

# PIML
plot_eclipse_behavior(piml, "Ecl_sun_pow_pay (PIML)")

# ML
plot_eclipse_behavior(ml, "Ecl_sun_pow_pay (ML)")
