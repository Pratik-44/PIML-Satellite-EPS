import pandas as pd
import matplotlib.pyplot as plt

# Load data
orig_path = "orig_mod_timecorr.csv"
orig = pd.read_csv(orig_path, parse_dates=["timestamp"])

'''
# Sort
orig = orig.sort_values("timestamp").reset_index(drop=True)
'''

# ===============================
# Heater Power Line Plot
# ===============================
plt.figure(figsize=(14,3))

plt.plot(orig["timestamp"],
         orig["heater_power"],
         linewidth=1,
         color="orange",
         label="Heater Power")

plt.title("Heater Power vs Time (Original Data)")
plt.ylabel("Heater Power (W)")
plt.xlabel("Time")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("heater_power.png", dpi=300)
plt.show()