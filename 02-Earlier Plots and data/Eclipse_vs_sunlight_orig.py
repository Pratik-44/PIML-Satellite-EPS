import pandas as pd
import matplotlib.pyplot as plt

orig_path = "orig_mod_timecorr.csv"

# Load CSV
orig = pd.read_csv(orig_path, parse_dates=["timestamp"])

'''
# Downsampling
step = max(len(orig) // 3000, 1)
orig = orig.iloc[::step]
'''

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

# Original
plot_eclipse_behavior(orig, "Eclipse vs Sunlight (Original)")
