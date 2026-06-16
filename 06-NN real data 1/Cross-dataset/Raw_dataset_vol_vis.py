import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATASET
# =========================

file_path = "F:/Downloads/PIML_Results/NN real data/training_data_synth.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset loaded:", df.shape)

# =========================
# 2. PLOT FUNCTION
# =========================

def plot_original_voltage(df, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=True
    )

    # =====================
    # Voltage plot (ONLY ORIGINAL)
    # =====================
    ax1.plot(
        df["timestamp"],
        df["voltage"],
        label="Original Voltage",
        color="blue",
        linewidth=1
    )

    # =====================
    # Eclipse shading
    # =====================
    eclipse = df["eclipse_flag"].values
    time = df["timestamp"].values

    in_eclipse = False
    start_time = None

    for i in range(len(df)):

        if eclipse[i] == 1 and not in_eclipse:
            in_eclipse = True
            start_time = time[i]

        elif eclipse[i] == 0 and in_eclipse:
            ax1.axvspan(
                start_time, time[i],
                color="gray", alpha=0.25
            )
            in_eclipse = False

    if in_eclipse:
        ax1.axvspan(
            start_time, time[-1],
            color="gray", alpha=0.25
        )

    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid()

    # =====================
    # Heater Power + Payload
    # =====================
    line1, = ax2.plot(
        df["timestamp"],
        df["heater_power"],
        color="green",
        linewidth=1,
        label="Heater Power"
    )

    ax2.set_ylabel("Heater Power (W)")
    ax2.grid()

    ax2_right = ax2.twinx()

    line2, = ax2_right.step(
        df["timestamp"],
        df["payload_status"],
        where="post",
        color="red",
        label="Payload Status"
    )

    ax2_right.set_ylabel("Payload Status")
    ax2_right.set_ylim(-0.05, 1.05)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


# =========================
# 3. PLOT DATASET
# =========================

plot_original_voltage(df, "Training_Data_synth_Original_Voltage_Profile")