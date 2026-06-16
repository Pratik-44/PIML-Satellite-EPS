import pandas as pd
import matplotlib.pyplot as plt

piml_path = "het_piml_out/long_rollout_12h_lstm_piml.csv"
ml_path   = "het_mlout/long_rollout_12h_lstm_ml.csv"

# Load CSVs
piml = pd.read_csv(piml_path, parse_dates=["timestamp"])
ml   = pd.read_csv(ml_path, parse_dates=["timestamp"])

'''
# Downsampling
step = max(len(piml) // 5000, 1)
piml = piml.iloc[::step]
ml   = ml.iloc[::step]
'''

import matplotlib.pyplot as plt

def plot_eclipse_behavior(df, title):
    plt.figure()
    plt.plot(df["timestamp"], df["total_power"], label="Total Power")

    # Shade eclipse regions
    for i in range(len(df)-1):
        if df["eclipse_flag"].iloc[i] == 1:
            plt.axvspan(df["timestamp"].iloc[i],
                        df["timestamp"].iloc[i+1],
                        color="gray", alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f"{title}.png")
    plt.show()

# PIML
plot_eclipse_behavior(piml, "Eclipse vs Sunlight (PIML)")

# ML
plot_eclipse_behavior(ml, "Eclipse vs Sunlight (ML)")
