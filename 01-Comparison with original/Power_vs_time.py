import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load files once
# -------------------------
orig = pd.read_csv("F:\Downloads\PIML_Results\orig_mod_timecorr.csv", parse_dates=["timestamp"])
piml = pd.read_csv("F:\Downloads\PIML_Results\het_piml_out\long_rollout_12h_lstm_piml.csv", parse_dates=["timestamp"])
ml   = pd.read_csv("F:\Downloads\PIML_Results\het_mlout\long_rollout_12h_lstm_ml.csv", parse_dates=["timestamp"])

# =========================
# POWER PLOTS
# =========================

# ---- ORIGINAL vs PIML (Power)
start_time = max(orig["timestamp"].min(), piml["timestamp"].min())
end_time   = min(orig["timestamp"].max(), piml["timestamp"].max())

orig_p = orig[(orig["timestamp"] >= start_time) & (orig["timestamp"] <= end_time)]
piml_p = piml[(piml["timestamp"] >= start_time) & (piml["timestamp"] <= end_time)]

step = max(len(orig_p) // 5000, 1)
orig_p = orig_p.iloc[::step]
piml_p = piml_p.iloc[::step]

plt.figure(figsize=(14,5))
plt.plot(orig_p["timestamp"], orig_p["total_power"], label="Original")
plt.plot(piml_p["timestamp"], piml_p["total_power"], label="PIML")

plt.xlabel("Time")
plt.ylabel("Total Power (W)")
plt.title("Total Power vs Time: Original vs PIML")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("power_original_vs_piml.png", dpi=300)
plt.show()


# ---- ORIGINAL vs ML (Power)
start_time = max(orig["timestamp"].min(), ml["timestamp"].min())
end_time   = min(orig["timestamp"].max(), ml["timestamp"].max())

orig_m = orig[(orig["timestamp"] >= start_time) & (orig["timestamp"] <= end_time)]
ml_m   = ml[(ml["timestamp"] >= start_time) & (ml["timestamp"] <= end_time)]

step = max(len(orig_m) // 5000, 1)
orig_m = orig_m.iloc[::step]
ml_m   = ml_m.iloc[::step]

plt.figure(figsize=(14,5))
plt.plot(orig_m["timestamp"], orig_m["total_power"], label="Original")
plt.plot(ml_m["timestamp"],   ml_m["total_power"], label="ML")

plt.xlabel("Time")
plt.ylabel("Total Power (W)")
plt.title("Total Power vs Time: Original vs ML")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("power_original_vs_ml.png", dpi=300)
plt.show()