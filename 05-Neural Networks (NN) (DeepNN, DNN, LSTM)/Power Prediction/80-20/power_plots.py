import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD PREDICTIONS
# =========================

dnn = pd.read_csv("power_DNN_predictions.csv")
deep = pd.read_csv("power_DeepNN_predictions.csv")
lstm = pd.read_csv("power_LSTM_predictions.csv")

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

    plt.title(f"Power: Original vs {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# =========================
# GENERATE ALL PLOTS
# =========================

plot_compare(dnn, "DNN", "power_vs_dnn.png")
plot_compare(deep, "DeepNN", "power_vs_deepnn.png")
plot_compare(lstm, "LSTM", "power_vs_lstm.png")

print("All power comparison plots saved.")