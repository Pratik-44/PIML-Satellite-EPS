import pandas as pd
import matplotlib.pyplot as plt

piml_path = "het_piml_out/long_rollout_12h_lstm_piml.csv"
ml_path   = "het_mlout/long_rollout_12h_lstm_ml.csv"

piml = pd.read_csv(piml_path, parse_dates=["timestamp"])
ml   = pd.read_csv(ml_path, parse_dates=["timestamp"])

step = max(len(piml) // 5000, 1)
piml = piml.iloc[::step]
ml   = ml.iloc[::step]

# -------------------------------
# 1. CURRENT vs TIME (Line Graph)
# -------------------------------
plt.figure(figsize=(12, 5))
plt.plot(piml["timestamp"], piml["current"], label="PIML", linewidth=2)
plt.plot(ml["timestamp"], ml["current"], label="ML", linewidth=1.5, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Current (A)")
plt.title("Current vs Time (PIML vs ML)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("current_vs_time.png")
plt.show()

# -------------------------------
# 2. VOLTAGE vs TIME (Line Graph)
# -------------------------------
plt.figure(figsize=(12, 5))
plt.plot(piml["timestamp"], piml["voltage"], label="PIML", linewidth=2)
plt.plot(ml["timestamp"], ml["voltage"], label="ML", linewidth=1.5, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Time (PIML vs ML)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("voltage_vs_time.png")
plt.show()

# -------------------------------
# 3. TOTAL POWER vs TIME (Line Graph)
# -------------------------------
plt.figure(figsize=(12, 5))
plt.plot(piml["timestamp"], piml["total_power"], label="PIML", linewidth=2)
plt.plot(ml["timestamp"], ml["total_power"], label="ML", linewidth=1.5, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Total Power (W)")
plt.title("Total Power vs Time (PIML vs ML)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("power_vs_time.png")
plt.show()

print("Line graphs generated and saved successfully.")
