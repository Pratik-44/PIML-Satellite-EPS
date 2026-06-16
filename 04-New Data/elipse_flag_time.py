import pandas as pd
import matplotlib.pyplot as plt

file_path = "samedata_r2.txt"

df = pd.read_csv(file_path)

# Remove trailing spaces in headers
df.columns = df.columns.str.strip()

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%d-%m-%Y-%H:%M:%S:%f"
)

# Sort by time
df = df.sort_values("Time").reset_index(drop=True)

# Create eclipse flag
# Raw: 1 = sunlit, 0 = eclipse
# Needed: 1 = eclipse, 0 = sunlight
df["eclipse_flag"] = (
    df["FDD08633:S/C_SUNLIT_ECL"] < 0.5
).astype(int)

plt.figure(figsize=(14,3))

plt.plot(
    df["Time"],
    df["eclipse_flag"],
    linewidth=1,
    label="Eclipse Flag"
)

plt.title("Eclipse Flag vs Time")
plt.xlabel("Time")
plt.ylabel("Eclipse (1) / Sunlight (0)")
plt.ylim(-0.1, 1.1)
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("Eclipse_Flag_LinePlot.png", dpi=300)
plt.show()
