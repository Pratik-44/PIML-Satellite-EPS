import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load files once
# -------------------------
orig = pd.read_csv("F:\Downloads\PIML_Results\orig_mod_timecorr.csv", parse_dates=["timestamp"])
piml = pd.read_csv("F:\Downloads\PIML_Results\het_piml_out\long_rollout_12h_lstm_piml.csv", parse_dates=["timestamp"])
ml   = pd.read_csv("F:\Downloads\PIML_Results\het_mlout\long_rollout_12h_lstm_ml.csv", parse_dates=["timestamp"])

# -------------------------
# ORIGINAL vs ML
# -------------------------
start_time = max(orig["timestamp"].min(), ml["timestamp"].min())
end_time   = min(orig["timestamp"].max(), ml["timestamp"].max())

orig_m = orig[(orig["timestamp"] >= start_time) & (orig["timestamp"] <= end_time)]
ml_m   = ml[(ml["timestamp"] >= start_time) & (ml["timestamp"] <= end_time)]

step = max(len(orig_m) // 5000, 1)
orig_m = orig_m.iloc[::step]
ml_m   = ml_m.iloc[::step]

fig, ax1 = plt.subplots(figsize=(14,5))

# -------- Current (left axis)
ax1.plot(orig_m["timestamp"], orig_m["current"],
         label="Original Current (A)", color="blue")

ax1.plot(ml_m["timestamp"], ml_m["current"],
         label="ML Current (A)", color="orange")

ax1.set_xlabel("Time")
ax1.set_ylabel("Current (A)")
ax1.grid(True)

# -------- Heater Power (right axis)
ax2 = ax1.twinx()

ax2.step(orig_m["timestamp"], orig_m["heater_power"],
         where="post", linestyle="--", color="blue",
         label="Original Heater (W)")

ax2.step(ml_m["timestamp"], ml_m["heater_power"],
         where="post", color="red",
         label="ML Heater (W)")

ax2.set_ylabel("Heater Power (W)")

# -------- Legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.title("Heater Influence: Original vs ML")
plt.tight_layout()
plt.savefig("heater_original_vs_ml2.png", dpi=300)
plt.show()

# -------------------------
# ORIGINAL vs PIML
# -------------------------
start_time = max(orig["timestamp"].min(), piml["timestamp"].min())
end_time   = min(orig["timestamp"].max(), piml["timestamp"].max())

orig_p = orig[(orig["timestamp"] >= start_time) & (orig["timestamp"] <= end_time)]
piml_p = piml[(piml["timestamp"] >= start_time) & (piml["timestamp"] <= end_time)]

step = max(len(orig_p) // 5000, 1)
orig_p = orig_p.iloc[::step]
piml_p = piml_p.iloc[::step]

fig, ax1 = plt.subplots(figsize=(14,5))

# -------- Current (left axis)
ax1.plot(orig_p["timestamp"], orig_p["current"],
         label="Original Current (A)", color="blue")

ax1.plot(piml_p["timestamp"], piml_p["current"],
         label="PIML Current (A)", color="orange")

ax1.set_xlabel("Time")
ax1.set_ylabel("Current (A)")
ax1.grid(True)

# -------- Heater Power (right axis)
ax2 = ax1.twinx()

ax2.step(orig_p["timestamp"], orig_p["heater_power"],
         where="post", linestyle="--", color="blue",
         label="Original Heater (W)")

ax2.step(piml_p["timestamp"], piml_p["heater_power"],
         where="post", color="red",
         label="PIML Heater (W)")

ax2.set_ylabel("Heater Power (W)")

# -------- Legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.title("Heater Influence: Original vs PIML")
plt.tight_layout()
plt.savefig("heater_original_vs_piml2.png", dpi=300)
plt.show()
