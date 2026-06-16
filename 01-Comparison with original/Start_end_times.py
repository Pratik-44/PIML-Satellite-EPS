import pandas as pd
import numpy as np

# Load files
orig = pd.read_csv("F:\Downloads\PIML_Results\orig_mod_timecorr.csv", parse_dates=["timestamp"])
piml = pd.read_csv("F:\Downloads\PIML_Results\het_piml_out\long_rollout_12h_lstm_piml.csv", parse_dates=["timestamp"])
ml   = pd.read_csv("F:\Downloads\PIML_Results\het_mlout\long_rollout_12h_lstm_ml.csv", parse_dates=["timestamp"])

# Sort by timestamp (important)
orig = orig.sort_values("timestamp").reset_index(drop=True)
piml = piml.sort_values("timestamp").reset_index(drop=True)
ml   = ml.sort_values("timestamp").reset_index(drop=True)

print("Original start:", orig["timestamp"].min())
print("PIML start    :", piml["timestamp"].min())
print("ML start      :", ml["timestamp"].min())

print("\nOriginal end  :", orig["timestamp"].max())
print("PIML end      :", piml["timestamp"].max())
print("ML end        :", ml["timestamp"].max())

"""
Original start: 2024-06-01 00:02:48.927000
PIML start    : 2024-06-01 00:02:48.927000
ML start      : 2024-06-01 00:02:48.927000

Original end  : 2024-06-01 18:39:24.891000
PIML end      : 2024-06-01 12:02:48.415000
ML end        : 2024-06-01 12:02:48.415000
"""