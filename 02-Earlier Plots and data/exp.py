import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Load data
# ===============================
orig_path = "orig_mod_timecorr.csv"
orig = pd.read_csv(orig_path, parse_dates=["timestamp"])

# sort by time
orig = orig.sort_values("timestamp").reset_index(drop=True)

#Force eclipse to clean binary
orig["eclipse_flag"] = (orig["eclipse_flag"] > 0.5).astype(int)

print(orig["eclipse_flag"].value_counts())
print("Transitions:", (orig["eclipse_flag"].diff() != 0).sum())
print(orig.groupby("eclipse_flag")["total_power"].mean())

# (optional) downsampling if needed
"""
step = max(len(orig) // 3000, 1)
orig = orig.iloc[::step]
"""

# ===============================
# Plot function
# ===============================
def plot_eclipse_behavior(df, title):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,7), sharex=True)

    # -------------------------
    # TOP: Total Power
    # -------------------------
    ax1.plot(df["timestamp"], df["total_power"],
             label="Total Power", color="blue")

    # ===== ROBUST ECLIPSE SHADING =====
    eclipse = df["eclipse_flag"].values
    time = df["timestamp"].values

    in_eclipse = False
    start_time = None

    for i in range(len(df)):

        # eclipse starts
        if eclipse[i] == 1 and not in_eclipse:
            in_eclipse = True
            start_time = time[i]

        # eclipse ends
        elif eclipse[i] == 0 and in_eclipse:
            end_time = time[i]
            ax1.axvspan(start_time, end_time,
                        color="gray", alpha=0.25)
            in_eclipse = False

    # handle case where file ends in eclipse
    if in_eclipse:
        ax1.axvspan(start_time, time[-1],
                    color="gray", alpha=0.25)

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


# ===============================
# CALL FUNCTION
# ===============================
plot_eclipse_behavior(orig, "exp")