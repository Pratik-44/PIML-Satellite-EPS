import pandas as pd

# Load dataset
df = pd.read_csv("F:/Downloads/PIML_Results/NN real data/feb2026rb2r2.txt")

# Convert timestamp column
df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format='%d-%m-%Y-%H:%M:%S:%f'
)

# Save updated dataset
df.to_csv("F:/Downloads/PIML_Results/NN real data/feb2026rb2r2_updated.txt", index=False)