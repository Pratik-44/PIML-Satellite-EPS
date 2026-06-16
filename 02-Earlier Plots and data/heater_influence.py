import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# File paths
# -------------------------------
piml_path = "het_piml_out/long_rollout_12h_lstm_piml.csv"
ml_path   = "het_mlout/long_rollout_12h_lstm_ml.csv"

# -------------------------------
# Load CSVs
# -------------------------------
piml = pd.read_csv(piml_path, parse_dates=["timestamp"])
ml   = pd.read_csv(ml_path, parse_dates=["timestamp"])

def plot_heater_influence(df, title):
    fig, ax1 = plt.subplots()

    # Current
    ax1.plot(df["timestamp"], df["current"], label="Current (A)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Current (A)")

    # Heater power
    ax2 = ax1.twinx()
    ax2.step(df["timestamp"], df["heater_power"],
             where="post", color="red", label="Heater Power (W)")
    ax2.set_ylabel("Heater Power (W)")

    fig.suptitle(title)
    fig.legend(loc="upper right")
    ax1.grid()
    plt.savefig(f"{title}.png")
    plt.show()

# PIML
plot_heater_influence(piml, "Heater Influence (PIML)")

# ML
plot_heater_influence(ml, "Heater Influence (ML)")
