#!/usr/bin/env python3
"""
integrated_piml_simulator.py

Unified simulator that supports:
 - C-RVAE (Conditional Recurrent VAE)  -> alias "crvae"
 - Heteroscedastic RNN (mean+var)     -> alias "hetero"
 - Physics-Informed LSTM (LSTM heads)  -> alias "lstm"

Features:
 - Physics-informed losses (energy balance, eclipse, heater, payload)
 - Safety constraints (voltage floor, no charging in eclipse, P_total <= V*I)
 - Payload penalty (L1 hinge)
 - Oversampling of payload windows
 - Long autoregressive rollout mode (--long_rollout_hours)
 - Output CSV of simulated telemetry

Author: Integrated version for user
"""

import os
import argparse
import math
import random
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='orig_mod_timecorr.csv', help='input CSV')
parser.add_argument('--model', type=str, choices=['crvae', 'hetero', 'lstm'], default='crvae')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--context_len', type=int, default=60)
parser.add_argument('--pred_horizon', type=int, default=60)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--lambda_phys', type=float, default=1.0)
parser.add_argument('--lambda_safety', type=float, default=1.0)
parser.add_argument('--beta_kl', type=float, default=1.0)
parser.add_argument('--lambda_payload', type=float, default=100.0)
parser.add_argument('--payload_power_nom', type=float, default=3500.0)
parser.add_argument('--payload_oversample_factor', type=int, default=20)
parser.add_argument('--long_rollout_hours', type=float, default=0.0)
parser.add_argument('--out_dir', type=str, default='outputs')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device(args.device)

# -----------------------
# Config: features & controls
# -----------------------
features = ['voltage', 'current', 'total_power']
controls = ['payload_status', 'eclipse_flag', 'heater_power']
feature_index_map = {n: i for i, n in enumerate(features)}
control_index_map = {n: i for i, n in enumerate(controls)}

# -----------------------
# Utility functions
# -----------------------
def dt_seconds_from_times(df):
    dts = df['timestamp'].diff().dt.total_seconds().dropna()
    return float(dts.median()) if len(dts) > 0 else 1.0

def soc_from_voc_lin(v, Vmin, Vmax):
    if Vmax == Vmin:
        return 0.5
    return np.clip((v - Vmin) / (Vmax - Vmin), 0.0, 1.0)

# -----------------------
# Data loading and preprocessing
# -----------------------
df_raw = pd.read_csv(args.input, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
dt_seconds = dt_seconds_from_times(df_raw)
Vmin = float(df_raw['voltage'].min())
Vmax = float(df_raw['voltage'].max())

# scalers
scaler_X = StandardScaler(); scaler_C = StandardScaler()
scaler_X.fit(df_raw[features].values)
scaler_C.fit(df_raw[controls].values)

# Dataset
class TelemetryDataset(Dataset):
    def __init__(self, df, ctx, pred, features, controls):
        self.df = df.reset_index(drop=True)
        self.ctx = ctx; self.pred = pred
        self.features = features; self.controls = controls
        self.indices = []
        N = len(df); L = ctx + pred
        for i in range(N - L + 1):
            self.indices.append(i)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        ctx_slice = self.df.iloc[i:i+self.ctx]
        pred_slice = self.df.iloc[i+self.ctx:i+self.ctx+self.pred]
        return {
            'x_ctx': ctx_slice[self.features].values.astype(np.float32),
            'x_fut': pred_slice[self.features].values.astype(np.float32),
            'c_ctx': ctx_slice[self.controls].values.astype(np.float32),
            'c_fut': pred_slice[self.controls].values.astype(np.float32),
            'idx': i
        }

dataset = TelemetryDataset(df_raw, args.context_len, args.pred_horizon, features, controls)

# Oversample payload windows
if args.payload_oversample_factor and args.payload_oversample_factor > 0:
    payload_windows = []
    for start_idx in list(dataset.indices):
        window = dataset.df.iloc[start_idx:start_idx+dataset.ctx+dataset.pred]
        if window['payload_status'].any():
            payload_windows.append(start_idx)
    if len(payload_windows) > 0:
        extra = []
        for idx0 in payload_windows:
            for _ in range(args.payload_oversample_factor):
                extra.append(idx0)
        dataset.indices.extend(extra)
        random.shuffle(dataset.indices)
        print(f"Oversampled {len(payload_windows)} payload windows, added {len(extra)} extra samples. New total windows: {len(dataset.indices)}")
    else:
        print("Warning: no payload windows found for oversampling.")

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# -----------------------
# Models
# -----------------------
# 1) C-RVAE (conditional recurrent VAE)
class CRVAE(nn.Module):
    def __init__(self, input_dim, control_dim, hidden_dim, latent_dim, out_dim):
        super().__init__()
        self.enc = nn.GRU(input_dim+control_dim, hidden_dim, batch_first=True)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.prior = nn.GRU(input_dim+control_dim, hidden_dim, batch_first=True)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec_proj = nn.Linear(input_dim+control_dim+latent_dim, hidden_dim)
        self.dec_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dec_head = nn.Linear(hidden_dim, out_dim)
    def encode_post(self, x_ctx, c_ctx, x_fut, c_fut):
        x = torch.cat([x_ctx, x_fut], dim=1)
        c = torch.cat([c_ctx, c_fut], dim=1)
        _, h = self.enc(torch.cat([x, c], dim=2)); h = h.squeeze(0)
        return self.enc_mu(h), self.enc_logvar(h)
    def encode_prior(self, x_ctx, c_ctx):
        _, h = self.prior(torch.cat([x_ctx, c_ctx], dim=2)); h = h.squeeze(0)
        return self.prior_mu(h), self.prior_logvar(h)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, x_ctx, c_ctx, c_fut, z, teacher_force=None):
        B = x_ctx.size(0)
        last = x_ctx[:, -1, :]
        h = torch.tanh(self.dec_proj(torch.cat([last, torch.zeros(B, c_fut.size(2), device=last.device)[:,0,:] if False else last*0, z], dim=1)))
        # simpler: init hidden from proj of last+z
        h0 = torch.tanh(self.dec_proj(torch.cat([last, z], dim=1))).unsqueeze(0)
        prev = last
        outs = []
        h_rnn = h0
        for t in range(c_fut.size(1)):
            inp = torch.cat([prev, c_fut[:, t, :], z], dim=1)
            inp_p = torch.tanh(self.dec_proj(inp)).unsqueeze(1)
            out, h_rnn = self.dec_rnn(inp_p, h_rnn)
            y = self.dec_head(out.squeeze(1))
            outs.append(y.unsqueeze(1))
            prev = teacher_force[:, t, :] if teacher_force is not None else y
        return torch.cat(outs, dim=1)
    def forward(self, x_ctx, c_ctx, x_fut, c_fut):
        q_mu, q_logvar = self.encode_post(x_ctx, c_ctx, x_fut, c_fut)
        p_mu, p_logvar = self.encode_prior(x_ctx, c_ctx)
        z = self.reparam(q_mu, q_logvar)
        y = self.decode(x_ctx, c_ctx, c_fut, z, teacher_force=x_fut)
        return y, (q_mu, q_logvar), (p_mu, p_logvar), z

# 2) Heteroscedastic RNN (mean + logvar)
class HeteroRNN(nn.Module):
    def __init__(self, in_dim, ctrl_dim, hid, out_dim):
        super().__init__()
        self.rnn = nn.GRU(in_dim+ctrl_dim, hid, batch_first=True)
        self.head = nn.Linear(hid, out_dim*2)
    def forward(self, x_ctx, c_ctx, c_fut):
        B = x_ctx.size(0); last = x_ctx[:, -1, :]; h_prev = torch.zeros(1, B, self.rnn.hidden_size, device=x_ctx.device)
        prev = last; means = []; logvars = []
        for t in range(c_fut.size(1)):
            inp = torch.cat([prev, c_fut[:, t, :]], dim=1).unsqueeze(1)
            out, h_prev = self.rnn(inp, h_prev)
            stats = self.head(out.squeeze(1)); m, lv = stats.chunk(2, dim=1)
            means.append(m.unsqueeze(1)); logvars.append(lv.unsqueeze(1))
            prev = m
        return torch.cat(means, 1), torch.cat(logvars, 1)

# 3) Physics-Informed LSTM (from orig_htr, adapted)
class SpacecraftPhysicsLaws:
    def __init__(self):
        self.critical_voltage = 35.0
        self.max_voltage = 43.0
        self.efficiency = 0.92
        self.payload_current_increase = 25.0
        self.payload_power_increase = 3500.0
        self.eclipse_heater_increase = 150.0
    def energy_conservation_loss(self, V, I, P):
        calc = V * I * self.efficiency
        err = torch.abs(calc - P) / (torch.abs(P) + 1e-6)
        return torch.mean(err)
    def eclipse_voltage_loss(self, V, eclipse_flag, prev_V):
        loss = torch.tensor(0.0, device=V.device)
        mask = (eclipse_flag == 0)
        if mask.sum() > 0:
            dV = V[mask] - prev_V[mask]
            loss = loss + torch.mean(torch.relu(dV))
        return loss
    def payload_power_loss(self, P, payload_flag, baseline, payload_nom):
        mask = (payload_flag > 0.5)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=P.device)
        inc = P[mask] - baseline[mask]
        # relative error to payload_nom
        err = torch.abs(inc - payload_nom) / (payload_nom + 1e-6)
        return torch.mean(err)
    def safety_loss(self, V):
        below = torch.relu(self.critical_voltage - V)
        over = torch.relu(V - self.max_voltage)
        return torch.mean(below)*100.0 + torch.mean(over)*10.0

class PhysicsInformedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.voltage_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2, 1))
        self.current_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2,1))
        self.power_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2,1))
        self.heater_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2,1))
        self.physics = SpacecraftPhysicsLaws()
    def forward(self, x):  # x: (B, T, input_size)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        v = self.voltage_head(last)
        i = self.current_head(last)
        p = self.power_head(last)
        h = self.heater_head(last)
        out = torch.cat([v, i, p, h], dim=1)
        return out
    def compute_losses(self, preds, targets, contexts, lambda_phys=1.0, payload_nom=3500.0):
        # preds: (B,4) [V,I,P,heater]
        V = preds[:,0]; I = preds[:,1]; P = preds[:,2]; H = preds[:,3]
        # targets expected as same shape if available
        mse = nn.MSELoss()(preds, targets)
        eclipse_flags = torch.tensor([c['eclipse_flag'] for c in contexts], dtype=torch.float32, device=preds.device)
        payload_flags = torch.tensor([c['payload_status'] for c in contexts], dtype=torch.float32, device=preds.device)
        prev_V = torch.tensor([c['prev_voltage'] for c in contexts], dtype=torch.float32, device=preds.device)
        baseline_power = torch.tensor([c.get('base_power', 0.0) for c in contexts], dtype=torch.float32, device=preds.device)
        phys_loss = 0.0
        phys_loss = phys_loss + self.physics.energy_conservation_loss(V, I, P)
        phys_loss = phys_loss + self.physics.eclipse_voltage_loss(V, eclipse_flags, prev_V)
        phys_loss = phys_loss + self.physics.payload_power_loss(P, payload_flags, baseline_power, payload_nom)
        phys_loss = phys_loss + self.physics.safety_loss(V)
        total_loss = mse + lambda_phys * phys_loss
        comp = {'data_loss': mse.item(), 'physics_loss': phys_loss.item() if isinstance(phys_loss, torch.Tensor) else float(phys_loss)}
        return total_loss, comp

# -----------------------
# Instantiate model
# -----------------------
if args.model == 'crvae':
    model = CRVAE(len(features), len(controls), args.hidden_dim, args.latent_dim, len(features)).to(device)
elif args.model == 'hetero':
    model = HeteroRNN(len(features), len(controls), args.hidden_dim, len(features)).to(device)
else:
    # LSTM input = features + controls
    input_size = len(features) + len(controls)
    model = PhysicsInformedLSTM(input_size, args.hidden_dim, num_layers=2, dropout=0.2).to(device)

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# -----------------------
# Loss helpers
# -----------------------
def kl_div(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q); var_p = torch.exp(logvar_p)
    return 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p)**2) / var_p - 1).sum(1).mean()

def physics_energy_loss(batch_idxes, df_raw, pred_tensor, ctrl_fut, dt, Vmin, Vmax, E_nom_wh=500.0, payload_power_nom=3500.0):
    # pred_tensor: (B,T,D) in original units (not scaled)
    B, T, D = pred_tensor.shape
    pred_total = pred_tensor[:,:,feature_index_map['total_power']].detach().cpu().numpy()
    pred_voltage = pred_tensor[:,:,feature_index_map['voltage']].detach().cpu().numpy()
    losses = []
    for b in range(B):
        idx0 = int(batch_idxes[b])
        win = df_raw.iloc[max(0, idx0-args.context_len): min(len(df_raw), idx0+args.context_len+args.pred_horizon)]
        local_max = win.loc[win['eclipse_flag']==1, 'total_power'].max()
        if np.isnan(local_max): local_max = df_raw['total_power'].median()
        P_solar_max = float(local_max)
        ctrl_seq = ctrl_fut[b].detach().cpu().numpy()
        eclipse_seq = ctrl_seq[:, control_index_map['eclipse_flag']]
        payload_seq = ctrl_seq[:, control_index_map['payload_status']]
        P_solar = (1 - eclipse_seq) * P_solar_max
        P_load = pred_total[b,:] + payload_seq * payload_power_nom
        P_net = P_solar - P_load
        deltaE_phys = np.sum(P_net) * dt / 3600.0
        V_seq = pred_voltage[b,:]
        soc_seq = np.array([soc_from_voc_lin(v, Vmin, Vmax) for v in V_seq])
        E_seq = soc_seq * E_nom_wh
        deltaE_model = E_seq[-1] - E_seq[0]
        losses.append((deltaE_model - deltaE_phys)**2)
    return torch.tensor(losses, dtype=torch.float32, device=pred_tensor.device).mean()

def safety_constraints_loss(pred_tensor, ctrl_fut, Vmin_hard=35.0, eff_max=1.0):
    # pred_tensor: (B,T,D) or (B,T) features
    V = pred_tensor[:,:,feature_index_map['voltage']]
    I = pred_tensor[:,:,feature_index_map['current']]
    P = pred_tensor[:,:,feature_index_map['total_power']]
    eclipse = ctrl_fut[:,:,control_index_map['eclipse_flag']]
    losses = []
    below = torch.relu(Vmin_hard - V); losses.append((below**2).mean())
    dV = V[:,1:] - V[:,:-1]; mask = eclipse[:,:-1] > 0.5
    if mask.any():
        dV_pos = torch.relu(dV[mask])
        if dV_pos.numel() > 0:
            losses.append((dV_pos**2).mean())
    VI = V * I
    excess = torch.relu(P - VI * eff_max); losses.append((excess**2).mean())
    return sum(losses)

# -----------------------
# Training loop
# -----------------------
print(f"Training model: {args.model} on device {device}")
for ep in range(1, args.epochs+1):
    model.train()
    epoch_loss = 0.0; n_batches = 0
    for batch_idx, batch in enumerate(loader):
        # Prepare inputs
        x_ctx = torch.tensor(scaler_X.transform(batch['x_ctx'].reshape(-1, len(features))).reshape(batch['x_ctx'].shape), dtype=torch.float32, device=device)
        x_fut = torch.tensor(scaler_X.transform(batch['x_fut'].reshape(-1, len(features))).reshape(batch['x_fut'].shape), dtype=torch.float32, device=device)
        c_ctx = torch.tensor(scaler_C.transform(batch['c_ctx'].reshape(-1, len(controls))).reshape(batch['c_ctx'].shape), dtype=torch.float32, device=device)
        c_fut = torch.tensor(scaler_C.transform(batch['c_fut'].reshape(-1, len(controls))).reshape(batch['c_fut'].shape), dtype=torch.float32, device=device)
        idxes = batch['idx'].numpy()
        opt.zero_grad()

        # Model-specific forward + losses
        if args.model == 'crvae':
            y_pred_scaled, (q_mu, q_lv), (p_mu, p_lv), z = model(x_ctx, c_ctx, x_fut, c_fut)
            # inverse scale to original units for physics/safety loss
            y_pred = torch.tensor(scaler_X.inverse_transform(y_pred_scaled.detach().cpu().numpy().reshape(-1, len(features))).reshape(y_pred_scaled.shape), dtype=torch.float32, device=device)
            y_true = torch.tensor(scaler_X.inverse_transform(x_fut.detach().cpu().numpy().reshape(-1, len(features))).reshape(x_fut.shape), dtype=torch.float32, device=device)
            rec_loss = nn.MSELoss()(y_pred, y_true)
            kld = kl_div(q_mu, q_lv, p_mu, p_lv)
            phys = physics_energy_loss(idxes, df_raw, y_pred, batch['c_fut'], dt_seconds, Vmin, Vmax, E_nom_wh=500.0, payload_power_nom=args.payload_power_nom)
            safe = safety_constraints_loss(y_pred, c_fut)
            loss = rec_loss + args.beta_kl * kld + args.lambda_phys * phys + args.lambda_safety * safe

        elif args.model == 'hetero':
            mu_scaled, lv_scaled = model(x_ctx, c_ctx, c_fut)
            mu = torch.tensor(scaler_X.inverse_transform(mu_scaled.detach().cpu().numpy().reshape(-1,len(features))).reshape(mu_scaled.shape), dtype=torch.float32, device=device)
            y_true = torch.tensor(scaler_X.inverse_transform(x_fut.detach().cpu().numpy().reshape(-1,len(features))).reshape(x_fut.shape), dtype=torch.float32, device=device)
            nll = nn.MSELoss()(mu, y_true)  # simplified NLL using MSE on mean
            phys = physics_energy_loss(idxes, df_raw, mu, batch['c_fut'], dt_seconds, Vmin, Vmax, E_nom_wh=500.0, payload_power_nom=args.payload_power_nom)
            safe = safety_constraints_loss(mu, c_fut)
            loss = nll + args.lambda_phys * phys + args.lambda_safety * safe

        else:  # lstm
            # build input: concat x_ctx last step features + controls over future? Use last context sequence + future controls appended
            # For simplicity, create input sequence = [ctx_window features + ctx_window controls]
            seq = np.concatenate([batch['x_ctx'], batch['c_ctx']], axis=2) if isinstance(batch['x_ctx'], np.ndarray) else None
            # Use scaled data: combine scaled X and scaled C
            Xs = torch.tensor(np.concatenate([batch['x_ctx'], batch['c_ctx']], axis=2), dtype=torch.float32, device=device)
            preds = model(Xs)  # returns (B,4) : [V,I,P,heater]
            # Build targets from x_fut first timestep (since LSTM modeled next-step)
            y_t = torch.tensor(np.stack([b[0] for b in batch['x_fut']]) , dtype=torch.float32, device=device) if False else None
            # create proper targets: we map features-> (V,I,P,heater) ordering; for compatibility convert x_fut[:,0,features]

            # --- Robust target extraction for LSTM branch ---
            # Convert batch arrays to numpy arrays (handles both list-of-arrays and already-array cases)
            x_fut_arr = np.asarray(batch['x_fut'])   # expected shape: (B, T, F)
            c_fut_arr = np.asarray(batch['c_fut'])   # expected shape: (B, T, C)

            # Defensive checks
            if x_fut_arr.ndim != 3:
                raise RuntimeError(f"Unexpected x_fut shape: {x_fut_arr.shape}, expected (B, T, F)")
            if c_fut_arr.ndim != 3:
                raise RuntimeError(f"Unexpected c_fut shape: {c_fut_arr.shape}, expected (B, T, C)")

            # Select the first future timestep as target (shape: B x F)
            first_fut = x_fut_arr[:, 0, :]   # (B, F)

            # Build targets: [voltage, current, total_power]
            targets_np = first_fut[:, [
                feature_index_map['voltage'],
                feature_index_map['current'],
                feature_index_map['total_power']
            ]]   # shape (B, 3)

            # Heater target from first future control timestep
            heater_targets_np = c_fut_arr[:, 0, control_index_map['heater_power']].reshape(-1, 1)  # shape (B,1)

            # Compose final target tensor (B, 4) = [V, I, P, heater]
            targets_full_np = np.concatenate([targets_np, heater_targets_np], axis=1)

            # Convert to torch tensors on device
            targets_full = torch.tensor(targets_full_np, dtype=torch.float32, device=device)
            # --- end robust extraction ---

            loss, comp = model.compute_losses(preds, targets_full, [{'eclipse_flag': int(cf[0,1]), 'payload_status': int(cf[0,0]), 'prev_voltage': float(batch['x_ctx'][i][-1,0]), 'base_power': float(np.median(df_raw['total_power']))} for i,cf in enumerate(batch['c_fut'])], lambda_phys=args.lambda_phys, payload_nom=args.payload_power_nom)

        # payload penalty (L1 hinge) - apply for CRVAE and hetero paths where y_pred exists
        if args.model in ('crvae', 'hetero'):
            # unscale controls
            c_fut_unscaled = scaler_C.inverse_transform(batch['c_fut'].reshape(-1, len(controls))).reshape(batch['c_fut'].shape)
            payload_seq = torch.tensor(c_fut_unscaled[:, :, control_index_map['payload_status']], dtype=torch.float32, device=device)
            if args.model == 'crvae':
                pred_P = y_pred[:, :, feature_index_map['total_power']]
            else:
                pred_P = mu
            baseline_load = float(df_raw.loc[df_raw['payload_status'] == 0, 'total_power'].median())
            expected_min = baseline_load + payload_seq * args.payload_power_nom
            payload_violation = torch.relu(torch.tensor(expected_min, dtype=pred_P.dtype, device=pred_P.device) - pred_P)
            payload_penalty = payload_violation.mean()
            loss = loss + args.lambda_payload * payload_penalty

        # backprop and step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        epoch_loss += float(loss.detach().cpu().numpy())
        n_batches += 1

    avg_loss = epoch_loss / max(1, n_batches)
    print(f"Epoch {ep}/{args.epochs}  avg_loss={avg_loss:.4f}")

# -----------------------
# Generation: long rollout or short test generation
# -----------------------
def generate_long_rollout(hours, model, df_raw, scaler_X, scaler_C, features, controls, device):
    steps = int(hours * 3600 / dt_seconds)
    print(f"Generating long rollout: {hours}h -> {steps} steps (dt={dt_seconds}s)")
    # seed context: last context_len rows
    x_ctx = df_raw[features].iloc[:args.context_len].values.astype(np.float32)[None,:,:]
    c_ctx = df_raw[controls].iloc[:args.context_len].values.astype(np.float32)[None,:,:]
    x_ctx_t = torch.tensor(scaler_X.transform(x_ctx.reshape(-1,len(features))).reshape(x_ctx.shape), dtype=torch.float32, device=device)
    c_ctx_t = torch.tensor(scaler_C.transform(c_ctx.reshape(-1,len(controls))).reshape(c_ctx.shape), dtype=torch.float32, device=device)

    # generate control rollout
    def gen_controls(steps):
        t = np.arange(steps) * dt_seconds
        orbit_period = 5400.0
        eclipse_flag = ((t % orbit_period) < (orbit_period * 0.35)).astype(float)
        payload_status = np.zeros(steps)
        # rare short bursts
        i=0
        while i<steps:
            if np.random.rand() < (1/1800)*dt_seconds:
                dur = max(1, int(5 / max(1.0, dt_seconds)))
                payload_status[i:i+dur] = 1.0
                i += dur
            else:
                i += 1
        heater_power = eclipse_flag * 150.0
        return np.stack([payload_status, eclipse_flag, heater_power], axis=1)
    controls_rollout = gen_controls(steps)[None,:,:]
    c_fut_t = torch.tensor(scaler_C.transform(controls_rollout.reshape(-1,len(controls))).reshape(controls_rollout.shape), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        if args.model == 'crvae':
            p_mu, p_lv = model.encode_prior(x_ctx_t, c_ctx_t)
            z = model.reparam(p_mu, p_lv)
            y_scaled = model.decode(x_ctx_t, c_ctx_t, c_fut_t, z, teacher_force=None)
            y = scaler_X.inverse_transform(y_scaled.cpu().numpy().reshape(-1, len(features))).reshape(y_scaled.shape)
        elif args.model == 'hetero':
            mean_scaled, lv = model(x_ctx_t, c_ctx_t, c_fut_t)
            y = scaler_X.inverse_transform(mean_scaled.cpu().numpy().reshape(-1, len(features))).reshape(mean_scaled.shape)
        else:
            # LSTM: roll step-by-step
            # create running window (scaled) of size context_len, each element has features+controls
            seq = np.concatenate([x_ctx, c_ctx], axis=2)[0]  # (ctx, F+C)
            y_out = np.zeros((1, steps, len(features)))
            for t in range(steps):
                input_seq = torch.tensor(seq[None,:,:], dtype=torch.float32, device=device)
                preds = model(input_seq)  # (1,4)
                v = preds[0,0].cpu().numpy(); i_ = preds[0,1].cpu().numpy(); p_ = preds[0,2].cpu().numpy()
                y_out[0,t,feature_index_map['voltage']] = v
                y_out[0,t,feature_index_map['current']] = i_
                y_out[0,t,feature_index_map['total_power']] = p_
                # shift seq: drop first, append new step (we need controls for next step)
                next_ctrl = controls_rollout[0,t,:]
                next_feat = np.zeros(len(features))
                next_row = np.concatenate([next_feat, next_ctrl])
                seq = np.vstack([seq[1:], next_row])
            y = y_out

    # runtime clamping & payload enforcement
    for t in range(y.shape[1]):
        Vt = float(y[0,t,feature_index_map['voltage']])
        It = float(y[0,t,feature_index_map['current']])
        Pt = float(y[0,t,feature_index_map['total_power']])
        # 1. voltage floor
        if Vt < 35.0:
            y[0,t,feature_index_map['voltage']] = 35.0
            Vt = 35.0
        # 2. no charging in eclipse
        if controls_rollout[0,t,control_index_map['eclipse_flag']] > 0.5 and t>0:
            if y[0,t,feature_index_map['voltage']] > y[0,t-1,feature_index_map['voltage']]:
                y[0,t,feature_index_map['voltage']] = y[0,t-1,feature_index_map['voltage']]
        # 3. power balance
        maxP = Vt * max(1e-6, It)
        if y[0,t,feature_index_map['total_power']] > maxP:
            y[0,t,feature_index_map['total_power']] = maxP
        # 4. payload enforcement
        if controls_rollout[0,t,control_index_map['payload_status']] > 0.5:
            expectedP = float(df_raw.loc[df_raw['payload_status']==0,'total_power'].median()) + args.payload_power_nom
            if y[0,t,feature_index_map['total_power']] < expectedP:
                y[0,t,feature_index_map['total_power']] = expectedP
            # recompute I and voltage drop (simple R_int model)
            Vt = float(y[0,t,feature_index_map['voltage']])
            It_new = y[0,t,feature_index_map['total_power']]/max(1e-6, Vt)
            R_int = 0.01
            Vnew = max(35.0, Vt - It_new * R_int)
            y[0,t,feature_index_map['voltage']] = Vnew
            y[0,t,feature_index_map['current']] = It_new

    # assemble dataframe
    rows = []
    start_time = df_raw['timestamp'].iloc[0]
    for t in range(y.shape[1]):
        row = {'timestamp': start_time + timedelta(seconds=float(t*dt_seconds))}
        for j, fname in enumerate(features):
            row[fname] = float(y[0,t,j])
        for j, cname in enumerate(controls):
            row[cname] = float(controls_rollout[0,t,j])
        rows.append(row)
    return pd.DataFrame(rows)

# call generation as requested
if args.long_rollout_hours > 0:
    out_df = generate_long_rollout(args.long_rollout_hours, model, df_raw, scaler_X, scaler_C, features, controls, device)
    out_csv = os.path.join(args.out_dir, f'long_rollout_{int(args.long_rollout_hours)}h_{args.model}.csv')
    out_df.to_csv(out_csv, index=False)
    print("Saved long rollout to", out_csv)
else:
    # short test generation using first window
    print("Generating short test rollout (pred_horizon)...")
    # use first context as seed
    x_ctx = df_raw[features].iloc[:args.context_len].values.astype(np.float32)[None,:,:]
    c_ctx = df_raw[controls].iloc[:args.context_len].values.astype(np.float32)[None,:,:]
    c_fut = df_raw[controls].iloc[args.context_len:args.context_len+args.pred_horizon].values.astype(np.float32)[None,:,:]
    x_ctx_t = torch.tensor(scaler_X.transform(x_ctx.reshape(-1,len(features))).reshape(x_ctx.shape), dtype=torch.float32, device=device)
    c_ctx_t = torch.tensor(scaler_C.transform(c_ctx.reshape(-1,len(controls))).reshape(c_ctx.shape), dtype=torch.float32, device=device)
    c_fut_t = torch.tensor(scaler_C.transform(c_fut.reshape(-1,len(controls))).reshape(c_fut.shape), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        if args.model == 'crvae':
            p_mu, p_lv = model.encode_prior(x_ctx_t, c_ctx_t)
            z = model.reparam(p_mu, p_lv)
            y_scaled = model.decode(x_ctx_t, c_ctx_t, c_fut_t, z, teacher_force=None)
            y = scaler_X.inverse_transform(y_scaled.cpu().numpy().reshape(-1,len(features))).reshape(y_scaled.shape)
        elif args.model == 'hetero':
            mean_scaled, lv = model(x_ctx_t, c_ctx_t, c_fut_t)
            y = scaler_X.inverse_transform(mean_scaled.cpu().numpy().reshape(-1,len(features))).reshape(mean_scaled.shape)
        else:
            seq = np.concatenate([x_ctx, c_ctx], axis=2)[0]
            # single-step rolling similar to long rollout
            steps = args.pred_horizon
            y_out = np.zeros((1, steps, len(features)))
            for t in range(steps):
                input_seq = torch.tensor(seq[None,:,:], dtype=torch.float32, device=device)
                preds = model(input_seq)
                v = preds[0,0].cpu().numpy(); i_ = preds[0,1].cpu().numpy(); p_ = preds[0,2].cpu().numpy()
                y_out[0,t,feature_index_map['voltage']] = v
                y_out[0,t,feature_index_map['current']] = i_
                y_out[0,t,feature_index_map['total_power']] = p_
                next_ctrl = c_fut[0,t,:]
                next_feat = np.zeros(len(features))
                next_row = np.concatenate([next_feat, next_ctrl])
                seq = np.vstack([seq[1:], next_row])
            y = y_out
    rows = []
    start = df_raw['timestamp'].iloc[0]
    for t in range(y.shape[1]):
        row = {'timestamp': start + timedelta(seconds=t*dt_seconds)}
        for j,f in enumerate(features):
            row[f] = float(y[0,t,j])
        for j,c in enumerate(controls):
            row[c] = float(c_fut[0,t,j])
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, f'simulated_short_{args.model}.csv')
    out_df.to_csv(out_csv, index=False)
    print("Saved short simulated output to", out_csv)

print("Done.")
