import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load files once
# -------------------------
orig = pd.read_csv("F:\Downloads\PIML_Results\orig_mod_timecorr.csv", parse_dates=["timestamp"])
piml = pd.read_csv("F:\Downloads\PIML_Results\het_piml_out\long_rollout_12h_lstm_piml.csv", parse_dates=["timestamp"])
ml   = pd.read_csv("F:\Downloads\PIML_Results\het_mlout\long_rollout_12h_lstm_ml.csv", parse_dates=["timestamp"])

# ORIGINAL vs PIML — Eclipse Behaviour
# Overlap trimming
start_time = max(orig["timestamp"].min(), piml["timestamp"].min())
end_time   = min(orig["timestamp"].max(), piml["timestamp"].max())

orig_p = orig[(orig["timestamp"] >= start_time) & (orig["timestamp"] <= end_time)]
piml_p = piml[(piml["timestamp"] >= start_time) & (piml["timestamp"] <= end_time)]

# Downsample
step = max(len(orig_p) // 5000, 1)
orig_p = orig_p.iloc[::step]
piml_p = piml_p.iloc[::step]

fig, ax1 = plt.subplots(figsize=(14,5))

# Power on left axis
ax1.plot(orig_p["timestamp"], orig_p["total_power"], label="Original Power")
ax1.plot(piml_p["timestamp"], piml_p["total_power"], label="PIML Power")
ax1.set_xlabel("Time")
ax1.set_ylabel("Total Power (W)")
ax1.grid(True)

# Eclipse flag on right axis
ax2 = ax1.twinx()
ax2.plot(orig_p["timestamp"], orig_p["eclipse_flag"],
         linestyle="--", alpha=0.4, label="Eclipse Flag")
ax2.set_ylabel("Eclipse Flag (0=Sunlight, 1=Eclipse)")
ax2.set_ylim(-0.1, 1.1)

fig.suptitle("Eclipse vs Sunlight (Power): Original vs PIML")
fig.tight_layout()
plt.savefig("eclipse_power_original_vs_piml.png", dpi=300)
plt.show()


# ORIGINAL vs ML — Eclipse Behaviour
# Overlap trimming
start_time = max(orig["timestamp"].min(), ml["timestamp"].min())
end_time   = min(orig["timestamp"].max(), ml["timestamp"].max())

orig_m = orig[(orig["timestamp"] >= start_time) & (orig["timestamp"] <= end_time)]
ml_m   = ml[(ml["timestamp"] >= start_time) & (ml["timestamp"] <= end_time)]

# Downsample
step = max(len(orig_m) // 5000, 1)
orig_m = orig_m.iloc[::step]
ml_m   = ml_m.iloc[::step]

fig, ax1 = plt.subplots(figsize=(14,5))

# Power on left axis
ax1.plot(orig_m["timestamp"], orig_m["total_power"], label="Original Power")
ax1.plot(ml_m["timestamp"], ml_m["total_power"], label="ML Power")
ax1.set_xlabel("Time")
ax1.set_ylabel("Total Power (W)")
ax1.grid(True)

# Eclipse flag on right axis
ax2 = ax1.twinx()
ax2.plot(orig_m["timestamp"], orig_m["eclipse_flag"],
         linestyle="--", alpha=0.4, label="Eclipse Flag")
ax2.set_ylabel("Eclipse Flag (0=Sunlight, 1=Eclipse)")
ax2.set_ylim(-0.1, 1.1)

fig.suptitle("Eclipse vs Sunlight (Power): Original vs ML")
fig.tight_layout()
plt.savefig("eclipse_power_original_vs_ml.png", dpi=300)
plt.show()
