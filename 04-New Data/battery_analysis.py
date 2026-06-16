import pandas as pd
import matplotlib.pyplot as plt

file_path = "samedata_r2.txt"

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

# timestamp
df["Time"] = pd.to_datetime(
    df["Time"],
    format="%d-%m-%Y-%H:%M:%S:%f"
)

# Sort by time
df = df.sort_values("Time").reset_index(drop=True)

# Eclipse flag
df["eclipse_flag"] = (
    df["FDD08633:S/C_SUNLIT_ECL"] < 0.5
).astype(int)

df["timestamp"] = df["Time"]
df["battery_current"] = df["PWRS0151:BAT_DISCHG_CUR_AMP"]
df["battery_voltage"] = df["PWRS0095:BAT_VOL_FINE_SEL_RT"]
df["payload_status"] = df["PLDHC041:PLC_M_EPC_STS"]
df["heater_power"] = df["THRS0001:HTR_LOAD_ON_BUS-A"]

def plot_battery_voltage_overlay(df, title):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14,7), sharex=True
    )

# Battery Current & Voltage
    ax1.plot(
        df["timestamp"],
        df["battery_current"],
        label="Battery Current (A)",
        color="blue",
        linewidth=1
    )

    ax1.plot(
        df["timestamp"],
        df["battery_voltage"],
        label="Battery Voltage (V)",
        color="green",
        linewidth=1
    )

# Eclipse shading
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
    # ax1.set_ylim(-1, 90)
    ax1.set_ylabel("Battery Current / Voltage")
    ax1.set_title(title)
    ax1.legend(loc = "upper right")
    ax1.grid()

    # Heater Power and Payload

    line_heater, = ax2.plot(
        df["timestamp"],
        df["heater_power"],
        color="orange",
        linewidth=1,
        label="Heater Power (W)"
    )

    ax2.set_ylabel("Heater Power (W)")
    ax2.grid()

    # right axis for payload status
    ax2r = ax2.twinx()

    line_payload, = ax2r.step(
        df["timestamp"],
        df["payload_status"],
        where="post",
        color="red",
        label="Payload Status"
    )

    ax2r.set_ylabel("Payload Status (0/1)")
    ax2r.set_ylim(-0.1, 1.1)

    # ----- Combined legend -----
    lines = [line_heater, line_payload]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()

plot_battery_voltage_overlay(df, "Battery_Voltage_Current_Eclipse_Analysis")

