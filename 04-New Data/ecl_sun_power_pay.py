import pandas as pd
import matplotlib.pyplot as plt

file_path = "samedata_r2.txt"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Parse custom timestamp
df["Time"] = pd.to_datetime(
    df["Time"],
    format="%d-%m-%Y-%H:%M:%S:%f"
)

# Sort by time
df = df.sort_values("Time").reset_index(drop=True)

# Create eclipse flag based on SUNLIT column
# SUNLIT column → convert to eclipse
df["eclipse_flag"] = (
    df["FDD08633:S/C_SUNLIT_ECL"] < 0.5
).astype(int)

# Check for eclipse = 0 and sunlight = 1
'''
print(df.groupby("FDD08633:S/C_SUNLIT_ECL")[["PWRS0152:SC_LOAD_PWR_W", "THRS0001:HTR_LOAD_ON_BUS-A"]].mean())
FDD08633:S/C_SUNLIT_ECL             PWRS0152:SC_LOAD_PWR_W     THRS0001:HTR_LOAD_ON_BUS-A
0.0                                  379.754659                  232.135669
1.0                                  408.797469                  223.097931
'''

# Rename key columns (inline with old logic)
df["total_power"] = df["PWRS0152:SC_LOAD_PWR_W"]
df["payload_status"] = df["PLDHC041:PLC_M_EPC_STS"]
df["heater_power"] = df["THRS0001:HTR_LOAD_ON_BUS-A"]
df["timestamp"] = df["Time"]

def plot_eclipse_behavior(df, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=True
    )

    # Total Power
    ax1.plot(
        df["timestamp"],
        df["total_power"],
        label="Total Power",
        color="blue"
    )

    # Eclipse 
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

    ax1.set_ylabel("Power (W)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid()

    # Payload Status
    ax2.step(
        df["timestamp"],
        df["payload_status"],
        where="post",
        color="red",
        label="Payload Status"
    )

    ax2.set_ylabel("Payload")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


# Heater line plot
def plot_heater(df):

    plt.figure(figsize=(14,3))

    plt.plot(
        df["timestamp"],
        df["heater_power"],
        color="orange",
        linewidth=1,
        label="Heater Power"
    )

    plt.title("Heater Power vs Time")
    plt.xlabel("Time")
    plt.ylabel("Heater Power (W)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("Heater_Power_LinePlot.png", dpi=300)
    plt.show()


# Run plots
plot_eclipse_behavior(df, "NewData_Eclipse_Behavior")
plot_heater(df)