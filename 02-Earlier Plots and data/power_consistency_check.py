import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
piml = pd.read_csv("het_piml_out/long_rollout_12h_lstm_piml.csv")
ml   = pd.read_csv("het_mlout/long_rollout_12h_lstm_ml.csv")

# Downsample for large datasets
step = max(len(piml) // 5000, 1)
piml = piml.iloc[::step]
ml   = ml.iloc[::step]

# Convert timestamp
piml["timestamp"] = pd.to_datetime(piml["timestamp"])
ml["timestamp"]   = pd.to_datetime(ml["timestamp"])

# Compute V*I
piml["calc_power"] = piml["voltage"] * piml["current"]
ml["calc_power"]   = ml["voltage"] * ml["current"]

# ---- PIML ----
plt.figure()
plt.plot(piml["timestamp"], piml["total_power"], label="PIML Total Power")
plt.plot(piml["timestamp"], piml["calc_power"], label="PIML V×I", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.title("Power Consistency Check (PIML)")
plt.legend()
plt.grid()
plt.savefig("Power Consistency Check (PIML).png")
plt.show()

# ---- ML ----
plt.figure()
plt.plot(ml["timestamp"], ml["total_power"], label="ML Total Power")
plt.plot(ml["timestamp"], ml["calc_power"], label="ML V×I", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Power (W)")
plt.title("Power Consistency Check (ML)")
plt.legend()
plt.grid()
plt.savefig("Power Consistency Check (ML).png")
plt.show()

print("Power consistency plots generated successfully.")