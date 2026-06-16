import pandas as pd
import numpy as np

# Load data
orig = pd.read_csv("F:\Downloads\PIML_Results\orig_mod_timecorr.csv", parse_dates=["timestamp"])
piml = pd.read_csv("F:\Downloads\PIML_Results\het_piml_out\long_rollout_12h_lstm_piml.csv", parse_dates=["timestamp"])
ml   = pd.read_csv("F:\Downloads\PIML_Results\het_mlout\long_rollout_12h_lstm_ml.csv", parse_dates=["timestamp"])

# Sort
orig = orig.sort_values("timestamp")
piml = piml.sort_values("timestamp")
ml   = ml.sort_values("timestamp")

# Restrict to overlapping window
start_time = max(orig["timestamp"].min(),
                 piml["timestamp"].min(),
                 ml["timestamp"].min())

end_time = min(orig["timestamp"].max(),
               piml["timestamp"].max(),
               ml["timestamp"].max())

orig = orig[(orig["timestamp"] >= start_time) & (orig["timestamp"] <= end_time)]
piml = piml[(piml["timestamp"] >= start_time) & (piml["timestamp"] <= end_time)]
ml   = ml[(ml["timestamp"] >= start_time) & (ml["timestamp"] <= end_time)]

# Interpolate model values onto original timestamps
piml_interp = pd.DataFrame()
ml_interp   = pd.DataFrame()

piml_interp["timestamp"] = orig["timestamp"]
ml_interp["timestamp"]   = orig["timestamp"]

for col in ["voltage", "current", "total_power"]:
    piml_interp[col] = np.interp(
        orig["timestamp"].astype(np.int64),
        piml["timestamp"].astype(np.int64),
        piml[col]
    )
    
    ml_interp[col] = np.interp(
        orig["timestamp"].astype(np.int64),
        ml["timestamp"].astype(np.int64),
        ml[col]
    )

# Error metrics function
def compute_metrics(true, pred):
    rmse = np.sqrt(np.mean((true - pred)**2))
    mae  = np.mean(np.abs(true - pred))
    corr = np.corrcoef(true, pred)[0,1]
    return rmse, mae, corr

# Compute for all three signals
results = {}

for col in ["voltage", "current", "total_power"]:
    rmse_p, mae_p, corr_p = compute_metrics(orig[col], piml_interp[col])
    rmse_m, mae_m, corr_m = compute_metrics(orig[col], ml_interp[col])
    
    results[col] = {
        "PIML": {"RMSE": rmse_p, "MAE": mae_p, "Corr": corr_p},
        "ML":   {"RMSE": rmse_m, "MAE": mae_m, "Corr": corr_m}
    }

# Print results
for col in results:
    print(f"\n==== {col.upper()} ====")
    print("PIML  -> RMSE:", results[col]["PIML"]["RMSE"],
          " MAE:", results[col]["PIML"]["MAE"],
          " Corr:", results[col]["PIML"]["Corr"])
    print("ML    -> RMSE:", results[col]["ML"]["RMSE"],
          " MAE:", results[col]["ML"]["MAE"],
          " Corr:", results[col]["ML"]["Corr"])

"""
==== VOLTAGE ====
PIML  -> RMSE: 3.807408061551064  MAE: 3.3646507249759336  Corr: 0.26792255235139406
ML    -> RMSE: 2.8933193340915278  MAE: 2.8156709407368474  Corr: 0.4074386113430808

==== CURRENT ====
PIML  -> RMSE: 22.023177929135677  MAE: 17.351253104469123  Corr: 0.2050348297879095
ML    -> RMSE: 15.679323974879312  MAE: 14.51988054150385  Corr: 0.19794125832175807

==== TOTAL_POWER ====
PIML  -> RMSE: 603.0595062467216  MAE: 414.51339028795513  Corr: -0.10129886459007123
ML    -> RMSE: 428.0457317556889  MAE: 320.7762763375139  Corr: -0.09637620461044304
"""