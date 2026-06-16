import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD PREDICTIONS
# =========================

dnn = pd.read_csv("voltage_DNN_predictions.csv")
deep = pd.read_csv("voltage_DeepNN_predictions.csv")
lstm = pd.read_csv("voltage_LSTM_predictions.csv")

# convert timestamp
for df in [dnn, deep, lstm]:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# optional zoom for clarity
N = 2000
dnn = dnn.iloc[:N]
deep = deep.iloc[:N]
lstm = lstm.iloc[:N]

# =========================
# PLOT FUNCTION
# =========================

def plot_compare(df, model_name, filename):
    plt.figure(figsize=(12,5))

    plt.plot(df["timestamp"], df["Actual"], label="Original", linewidth=2)
    plt.plot(df["timestamp"], df["Predicted"], label=model_name)

    plt.title(f"Voltage: Original vs {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# =========================
# GENERATE ALL PLOTS
# =========================

plot_compare(dnn, "DNN", "voltage_vs_dnn.png")
plot_compare(deep, "DeepNN", "voltage_vs_deepnn.png")
plot_compare(lstm, "LSTM", "voltage_vs_lstm.png")

print("All voltage comparison plots saved.")