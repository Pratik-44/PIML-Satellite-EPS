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

# Original
plot_eclipse_behavior(orig, "Ecl_sun_pow_pay (Original)")
