import pandas as pd
import matplotlib.pyplot as plt

# Load data
orig_path = "orig_mod_timecorr.csv"
orig = pd.read_csv(orig_path, parse_dates=["timestamp"])

# Sort (important)
orig = orig.sort_values("timestamp").reset_index(drop=True)

# Clean to strict binary
orig["eclipse_flag"] = (orig["eclipse_flag"] > 0.5).astype(int)

# Eclipse line plot
plt.figure(figsize=(14,3))

plt.plot(orig["timestamp"],
         orig["eclipse_flag"],
         linewidth=1)

plt.title("Eclipse Flag vs Time (Original Data)")
plt.ylabel("Eclipse Flag")
plt.xlabel("Time")
plt.ylim(-0.1, 1.1)
plt.grid()

plt.tight_layout()
plt.savefig("Original_Eclipse_LinePlot.png", dpi=300)
plt.show()