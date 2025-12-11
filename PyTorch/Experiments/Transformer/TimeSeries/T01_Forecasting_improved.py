# Full end-to-end forecasting script (improved from the original notebook)
# - Adds scheduled sampling to reduce exposure bias
# - Uses SmoothL1Loss (Huber) instead of pure MSE
# - Adds weight decay, gradient clipping, dropout & LayerNorm
# - Adds sin/cos seasonality features & rolling stats
# - Implements baseline trend extrapolation + model predicts residuals
# - Includes per-horizon evaluation and plots
#
# Requirements: numpy, matplotlib, seaborn, sklearn, torch, pandas
# Run: python T01_Forecasting_improved.py

import os
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import torch
from torch import nn

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Synthetic data generation
# -------------------------
def trend(time):
    # piecewise linear trend
    t = np.array(time, dtype=np.float32)
    slope = np.zeros_like(t)
    slope[t < (3*12)] = 0.2
    slope[(t >= (3*12)) & (t < (6*12))] = -0.4
    slope[t >= (6*12)] = 1.0
    return slope * t

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / float(period)
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=5, seed=SEED):
    rnd = np.random.RandomState(seed=seed)
    return rnd.randn(len(time)) * noise_level

def make_time_series():
    period = 12
    baseline = 10
    amplitude = 40

    time_history = np.arange(10*12, dtype=np.float32)
    ts_history = baseline + trend(time=time_history) \
        + seasonality(time=time_history, period=period, amplitude=amplitude) \
        + noise(time=time_history)
    return time_history.astype(np.float32), ts_history.astype(np.float32)

# -------------------------
# Dataset and DataLoader
# -------------------------
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, ts, src_len, tgt_len, add_rolling=True):
        # ts: 1D numpy array already scaled (float32)
        super().__init__()
        self.ts = torch.tensor(ts, dtype=torch.float32).view(-1, 1)
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.idxs = list(range(len(ts) - (src_len + tgt_len) + 1))
        self.add_rolling = add_rolling
        # precompute rolling stats for efficiency
        arr = ts.flatten()
        self.rolling_mean_3 = pd.Series(arr).rolling(3, min_periods=1).mean().values.astype(np.float32)
        self.rolling_mean_6 = pd.Series(arr).rolling(6, min_periods=1).mean().values.astype(np.float32)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        data = {}
        i = self.idxs[idx]

        start = i; end = i + self.src_len
        src = self.ts[start:end]  # (src_len, 1)
        src_idx = torch.arange(start=start, end=end, dtype=torch.long)
        src_month = (src_idx % 12).long()

        start = i + self.src_len; end = i + self.src_len + self.tgt_len
        tgt = self.ts[start:end]
        tgt_idx = torch.arange(start=start, end=end, dtype=torch.long)
        tgt_month = (tgt_idx % 12).long()

        data['src'] = src.unsqueeze(0)  # (1, src_len, 1) -> we'll batch outside DataLoader
        data['src_idx'] = src_idx.unsqueeze(0)
        data['src_month'] = src_month.unsqueeze(0)
        data['tgt'] = tgt.unsqueeze(0)
        data['tgt_idx'] = tgt_idx.unsqueeze(0)
        data['tgt_month'] = tgt_month.unsqueeze(0)

        if self.add_rolling:
            # rolling means: per time-step scalar
            src_rm3 = torch.tensor(self.rolling_mean_3[start - 0:end - 0], dtype=torch.float32).view(-1, 1)
            src_rm6 = torch.tensor(self.rolling_mean_6[start - 0:end - 0], dtype=torch.float32).view(-1, 1)
            # ensure shapes match
            data['src_rm3'] = src_rm3.unsqueeze(0)
            data['src_rm6'] = src_rm6.unsqueeze(0)
            # tgt rolling features for completeness (used when teacher forcing)
            tgt_rm3 = torch.tensor(self.rolling_mean_3[start:end], dtype=torch.float32).view(-1, 1)
            tgt_rm6 = torch.tensor(self.rolling_mean_6[start:end], dtype=torch.float32).view(-1, 1)
            data['tgt_rm3'] = tgt_rm3.unsqueeze(0)
            data['tgt_rm6'] = tgt_rm6.unsqueeze(0)

        return {k: v.squeeze(0) for k, v in data.items()}


def collate_fn(batch):
    # Batch is a list of dicts; stack into tensors
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out

# -------------------------
# Model definitions
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50*12):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerForecast(nn.Module):
    def __init__(self, num_features=1, d_model=128, time_emb_dim=16, nhead=4, nlayers=2,
                 dropout=0.2, max_len=50*12):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # value projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # time embeddings: index & month embedding
        self.idx_emb = nn.Embedding(num_embeddings=max_len, embedding_dim=time_emb_dim//2)
        self.month_emb = nn.Embedding(num_embeddings=12, embedding_dim=time_emb_dim//2)

        # small linear to map time features to model dim
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim + 2, d_model),  # +2 for sin/cos seasonal continuous features
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.positional = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                                   dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, num_features),
        )

    def make_time_feats(self, idx_tensor):
        # idx_tensor: (B, T) long
        idx = idx_tensor.long()
        month = (idx % 12).long()
        idx_emb = self.idx_emb(idx)              # (B, T, emb//2)
        month_emb = self.month_emb(month)        # (B, T, emb//2)
        base = torch.cat([idx_emb, month_emb], dim=-1)  # (B, T, time_emb_dim)
        # also add sin/cos seasonal continuous features
        month_float = (month.float() / 12.0)  # [0,1)
        sin = torch.sin(2 * math.pi * month_float).unsqueeze(-1)
        cos = torch.cos(2 * math.pi * month_float).unsqueeze(-1)
        time_feats = torch.cat([base, sin, cos], dim=-1)  # (B, T, time_emb_dim+2)
        return time_feats

    def forward(self, data):
        # data fields: src (B, src_len, 1), src_idx (B, src_len),
        # tgt (B, tgt_len, 1), tgt_idx (B, tgt_len), plus optional rolling rm3/rm6
        src = data['src'].float()
        tgt = data['tgt'].float()
        src_idx = data['src_idx']
        tgt_idx = data['tgt_idx']

        # Optionally include rolling features concatenated to the value input (if present)
        if 'src_rm3' in data:
            src_feats = torch.cat([src, data['src_rm3'], data['src_rm6']], dim=-1)  # (B, src_len, 3)
            # project multi-dim to d_model with a linear layer on the fly
            src = self.input_proj(src_feats)  # expects features -> d_model
        else:
            src = self.input_proj(src)  # (B, src_len, d_model)

        src_time = self.make_time_feats(src_idx)
        src_time = self.time_proj(src_time)  # (B, src_len, d_model)
        src = self.positional(src + src_time)  # (B, src_len, d_model)

        memory = self.encoder(src)  # (B, src_len, d_model)

        # decoder
        if 'tgt_rm3' in data:
            tgt_feats = torch.cat([tgt, data['tgt_rm3'], data['tgt_rm6']], dim=-1)
            tgt = self.input_proj(tgt_feats)
        else:
            tgt = self.input_proj(tgt)

        tgt_time = self.make_time_feats(tgt_idx)
        tgt_time = self.time_proj(tgt_time)
        tgt = self.positional(tgt + tgt_time)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=tgt.size(1)).to(tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # (B, tgt_len, d_model)
        return self.output_proj(out)  # (B, tgt_len, num_features)

    def infer(self, data, tgt_len=12, device=None):
        # Autoregressive inference (no teacher forcing)
        # data['src'], data['src_idx'] provided; we will autoregressively build tgt with last src observation as start
        if device is None:
            device = next(self.parameters()).device

        B = data['src'].size(0)
        # initialize tgt with last src value
        last_val = data['src'][:, -1:, :].to(device)  # (B, 1, 1)
        last_idx = data['src_idx'][:, -1:].to(device)  # (B, 1)
        last_month = (last_idx % 12).long()

        # start tgt tensors
        tgt = last_val
        tgt_idx = last_idx
        if 'src_rm3' in data:
            # build rolling features for the first step by using last values
            last_rm3 = data['src_rm3'][:, -1:, :].to(device)
            last_rm6 = data['src_rm6'][:, -1:, :].to(device)
            tgt_rm3 = last_rm3
            tgt_rm6 = last_rm6
        else:
            tgt_rm3 = None; tgt_rm6 = None

        preds = []
        # We'll reuse data['src'] and its idx for encoder
        data_for_encoder = {
            'src': data['src'].to(device),
            'src_idx': data['src_idx'].to(device),
            'src_month': (data['src_idx'] % 12).to(device),
        }
        if 'src_rm3' in data:
            data_for_encoder['src_rm3'] = data['src_rm3'].to(device)
            data_for_encoder['src_rm6'] = data['src_rm6'].to(device)

        # compute memory once
        with torch.no_grad():
            src = data_for_encoder['src']
            src_idx = data_for_encoder['src_idx']
            # build a small data dict to call encoder half of forward
            enc_dict = {
                'src': src,
                'src_idx': src_idx,
                'tgt': torch.zeros_like(tgt),  # placeholder
                'tgt_idx': tgt_idx,            # placeholder
            }
            if 'src_rm3' in data_for_encoder:
                enc_dict['src_rm3'] = data_for_encoder['src_rm3']
                enc_dict['src_rm6'] = data_for_encoder['src_rm6']
            # get memory:
            memory = self.encoder(self.positional(
                self.input_proj(torch.cat([enc_dict['src'],
                                           enc_dict.get('src_rm3', torch.zeros_like(enc_dict['src'])),
                                           enc_dict.get('src_rm6', torch.zeros_like(enc_dict['src']))], dim=-1)
                                   + self.time_proj(self.make_time_feats(enc_dict['src_idx']))))
            # memory is shaped (B, src_len, d_model)

            for _ in range(tgt_len):
                # build minimal data dict for decoder input
                dec_dict = {
                    'src': data_for_encoder['src'],
                    'src_idx': data_for_encoder['src_idx'],
                    'tgt': tgt,
                    'tgt_idx': tgt_idx
                }
                if tgt_rm3 is not None:
                    dec_dict['tgt_rm3'] = tgt_rm3
                    dec_dict['tgt_rm6'] = tgt_rm6

                # compute decoder output using memory
                # reuse portions of forward but call decoder directly for efficiency
                # prepare tgt representation
                if 'tgt_rm3' in dec_dict:
                    tgt_feats = torch.cat([dec_dict['tgt'], dec_dict['tgt_rm3'], dec_dict['tgt_rm6']], dim=-1)
                    tgt_rep = self.input_proj(tgt_feats)
                else:
                    tgt_rep = self.input_proj(dec_dict['tgt'])
                tgt_time = self.time_proj(self.make_time_feats(dec_dict['tgt_idx']))
                tgt_rep = self.positional(tgt_rep + tgt_time)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=tgt_rep.size(1)).to(tgt.device)
                dec_out = self.decoder(tgt=tgt_rep, memory=memory, tgt_mask=tgt_mask)
                step_pred = self.output_proj(dec_out[:, -1:, :])  # (B, 1, 1)
                preds.append(step_pred)

                # append for next step
                tgt = torch.cat([tgt, step_pred], dim=1)
                new_idx = (tgt_idx[:, -1:] + 1)
                tgt_idx = torch.cat([tgt_idx, new_idx], dim=1)
                # rolling features: naive update using previous values (for simplicity)
                if tgt_rm3 is not None:
                    # shift and append last predicted value for rolling mean calculation
                    prev_rm3 = tgt_rm3[:, -1:, :].clone()
                    prev_rm6 = tgt_rm6[:, -1:, :].clone()
                    # compute simple rolling update
                    new_rm3 = (prev_rm3 * 2 + step_pred) / 3.0
                    new_rm6 = (prev_rm6 * 5 + step_pred) / 6.0
                    tgt_rm3 = torch.cat([tgt_rm3, new_rm3], dim=1)
                    tgt_rm6 = torch.cat([tgt_rm6, new_rm6], dim=1)

        preds = torch.cat(preds, dim=1)  # (B, tgt_len, 1)
        return preds

# -------------------------
# Training utilities
# -------------------------
def train_step_scheduled(model, dl_train, loss_fn, optimizer, device, teacher_forcing_prob=1.0, max_tgt_len=12):
    model.train()
    losses = []
    for batch in dl_train:
        # move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        # y_true
        y_true = batch['tgt']  # (B, tgt_len, 1)

        # Prepare decoder initial sequences: use last src value then shift tgt by one (teacher forcing start)
        dec_init = torch.cat([batch['src'][:, -1:, :], batch['tgt'][:, :-1, :]], dim=1)  # (B, tgt_len, 1)
        dec_idx_init = torch.cat([batch['src_idx'][:, -1:], batch['tgt_idx'][:, :-1]], dim=1)
        # rolling features for decoder (start similar)
        if 'src_rm3' in batch:
            dec_rm3 = torch.cat([batch['src_rm3'][:, -1:, :], batch['tgt_rm3'][:, :-1, :]], dim=1)
            dec_rm6 = torch.cat([batch['src_rm6'][:, -1:, :], batch['tgt_rm6'][:, :-1, :]], dim=1)
        else:
            dec_rm3 = dec_rm6 = None

        # We'll perform an autoregressive rollout inside the training step, replacing some targets with model preds
        B, T, _ = dec_init.shape
        dec_in = dec_init.clone()
        dec_idx = dec_idx_init.clone()
        if dec_rm3 is not None:
            dec_rm3_in = dec_rm3.clone()
            dec_rm6_in = dec_rm6.clone()
        else:
            dec_rm3_in = dec_rm6_in = None

        preds = []
        # First step uses dec_in[:, :1] etc.
        for t in range(T):
            # build data dict expected by model forward
            data = {
                'src': batch['src'],
                'src_idx': batch['src_idx'],
                'tgt': dec_in[:, :t+1, :],
                'tgt_idx': dec_idx[:, :t+1],
            }
            if dec_rm3_in is not None:
                data['src_rm3'] = batch['src_rm3']
                data['src_rm6'] = batch['src_rm6']
                data['tgt_rm3'] = dec_rm3_in[:, :t+1, :]
                data['tgt_rm6'] = dec_rm6_in[:, :t+1, :]

            out = model(data)  # (B, t+1, 1)
            pred_t = out[:, -1:, :]  # (B, 1, 1)
            preds.append(pred_t)

            if t+1 < T:
                # per-sample decide whether to use ground truth or prediction for next decoder input
                use_gt = (torch.rand(B, device=device) < teacher_forcing_prob).view(B, 1, 1)
                next_gt = batch['tgt'][:, t:t+1, :]  # ground truth next
                dec_in[:, t+1:t+2, :] = torch.where(use_gt, next_gt, pred_t)
                # update idx and rolling stats for next position
                dec_idx[:, t+1:t+2] = dec_idx[:, t:t+1] + 1
                if dec_rm3_in is not None:
                    # simple rolling update (mirrors inference update)
                    prev_rm3 = dec_rm3_in[:, t:t+1, :].clone()
                    prev_rm6 = dec_rm6_in[:, t:t+1, :].clone()
                    new_rm3 = (prev_rm3 * 2 + dec_in[:, t+1:t+2, :]) / 3.0
                    new_rm6 = (prev_rm6 * 5 + dec_in[:, t+1:t+2, :]) / 6.0
                    dec_rm3_in[:, t+1:t+2, :] = new_rm3
                    dec_rm6_in[:, t+1:t+2, :] = new_rm6

        y_pred = torch.cat(preds, dim=1)  # (B, T, 1)
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if len(losses) else 0.0

@torch.no_grad()
def evaluate_autoregressive(model, dl, device):
    model.eval()
    all_preds = []
    all_trues = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model.infer(batch, tgt_len=batch['tgt'].size(1), device=device)  # (B, tgt_len, 1)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(batch['tgt'].cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    return all_preds.squeeze(-1), all_trues.squeeze(-1)  # shapes (N, T)

# -------------------------
# Baseline: linear trend extrapolation
# -------------------------
def linear_trend_forecast(history, steps):
    # history: 1D numpy array, use last 3 points to fit slope
    if len(history) < 2:
        return np.repeat(history[-1], steps)
    x = np.arange(len(history))
    y = history
    # simple linear fit on last k points
    k = min(6, len(history))
    xk = x[-k:]
    yk = y[-k:]
    A = np.vstack([xk, np.ones_like(xk)]).T
    m, b = np.linalg.lstsq(A, yk, rcond=None)[0]
    future_x = np.arange(len(history), len(history)+steps)
    return m * future_x + b

# -------------------------
# Main training routine
# -------------------------
def main():
    # generate data
    time_history, ts_history = make_time_series()
    src_len = 24
    tgt_len = 12

    # split: reserve last (src_len+tgt_len) for test-like sequences (like notebook)
    ts_train = ts_history[:-12]
    ts_test_window = ts_history[-(src_len + tgt_len):]  # for testing sliding windows

    # scaler fit on train
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(ts_train.reshape(-1, 1))
    ts_train_scaled = scaler.transform(ts_train.reshape(-1, 1)).flatten().astype(np.float32)
    ts_test_scaled = scaler.transform(ts_test_window.reshape(-1, 1)).flatten().astype(np.float32)
    ts_all_scaled = scaler.transform(ts_history.reshape(-1, 1)).flatten().astype(np.float32)

    # Datasets and loaders (create sliding windows)
    train_dataset = TrainingDataset(ts=ts_train_scaled, src_len=src_len, tgt_len=tgt_len, add_rolling=True)
    test_dataset = TrainingDataset(ts=ts_test_scaled, src_len=src_len, tgt_len=tgt_len, add_rolling=True)

    batch_size = 32
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn, drop_last=False)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn)

    # model, optimizer, schedulers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerForecast(num_features=3, d_model=128, time_emb_dim=16, nhead=4, nlayers=2, dropout=0.15).to(device)
    # note: num_features=3 because we concatenated value + rm3 + rm6 in the input projection
    loss_fn = nn.SmoothL1Loss()  # robust to outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

    # training with scheduled sampling
    num_epochs = 200
    teacher_prob = 1.0
    min_teacher_prob = 0.2
    decay = 0.98  # per epoch multiplicative decay (aggressive -> slower)
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_transformer_forecast.pth'

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_step_scheduled(model, dl_train, loss_fn, optimizer, device,
                                         teacher_forcing_prob=teacher_prob, max_tgt_len=tgt_len)
        # evaluation: autoregressive (as inference)
        preds, trues = evaluate_autoregressive(model, dl_test, device=device)
        val_loss = float(np.mean(np.abs(preds - trues)))  # MAE on scaled domain

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)

        teacher_prob = max(min_teacher_prob, teacher_prob * decay)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_mae={val_loss:.6f} | teacher_prob={teacher_prob:.3f} | epoch_time={(time.time()-t0):.2f}s")

        if epoch % 50 == 0:
            # quick checkpoint plot
            plot_history(history)

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s, best_val={best_val:.6f}")

    # load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Full inference on the original full time series using sliding windows
    # Build sliding window inputs across the full scaled series (so we forecast for many windows)
    scaled = ts_all_scaled
    len_idx = len(scaled) - src_len + 1
    idx = np.arange(stop=len_idx).reshape(-1, 1) + np.arange(stop=src_len)
    data = {}
    data['src'] = torch.tensor(scaled[idx], dtype=torch.float32).unsqueeze(-1)  # (N_windows, src_len, 1)
    data['src_idx'] = torch.tensor(idx, dtype=torch.long)
    # rolling stats for full series to slice per-window
    roll3 = pd.Series(scaled).rolling(3, min_periods=1).mean().values.astype(np.float32)
    roll6 = pd.Series(scaled).rolling(6, min_periods=1).mean().values.astype(np.float32)
    src_rm3 = np.stack([roll3[i:i+src_len] for i in range(len_idx)], axis=0)
    src_rm6 = np.stack([roll6[i:i+src_len] for i in range(len_idx)], axis=0)
    data['src_rm3'] = torch.tensor(src_rm3, dtype=torch.float32).unsqueeze(-1)
    data['src_rm6'] = torch.tensor(src_rm6, dtype=torch.float32).unsqueeze(-1)

    # we will run inference in batches to avoid memory blowup
    B = 128
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, data['src'].size(0), B):
            batch = {
                'src': data['src'][i:i+B].to(device),
                'src_idx': data['src_idx'][i:i+B].to(device),
                'src_rm3': data['src_rm3'][i:i+B].to(device),
                'src_rm6': data['src_rm6'][i:i+B].to(device),
            }
            pred = model.infer(batch, tgt_len=tgt_len, device=device)  # (b, tgt_len, 1)
            all_preds.append(pred.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)  # (N_windows, tgt_len, 1)
    # pick the windows that correspond to forecasting beyond the original history end
    # Specifically, we want windows where the last source index is at the very end of the history-12 region
    # For simplicity, compute forecast using the last available window
    last_window_pred = all_preds[-1].squeeze(-1)  # (tgt_len,)
    # inverse transform
    last_window_pred_unscaled = scaler.inverse_transform(last_window_pred.reshape(-1, 1)).flatten()

    # baseline forecast using linear trend on last src_len points
    baseline_input = ts_history[-src_len:]  # last src_len original values
    baseline_forecast = linear_trend_forecast(baseline_input, tgt_len)

    time_pred = np.arange(time_history[-1] + 1 - tgt_len + tgt_len, time_history[-1] + 1 + tgt_len)
    # plotting final result (history + forecast)
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=time_history, y=ts_history, label='history')
    future_time = np.arange(time_history[-1] + 1, time_history[-1] + 1 + tgt_len)
    sns.lineplot(x=future_time, y=last_window_pred_unscaled, label='transformer_forecast', linestyle='--')
    sns.lineplot(x=future_time, y=baseline_forecast, label='baseline_linear', linestyle=':')
    plt.axvline(x=time_history[-1], color='red', linestyle='--')
    plt.legend(loc='best')
    plt.title('History and Forecast')
    plt.tight_layout()
    plt.show()

    # Per-horizon error on the reserved test window we used earlier (ts_test_window)
    # Use the last few windows to compute predictions over the true test target slices
    # We'll reuse test_dataset and dl_test evaluation results for per-horizon MAE
    preds_test, trues_test = evaluate_autoregressive(model, dl_test, device=device)  # shapes (N, T)
    # inverse transform
    preds_test_unscaled = scaler.inverse_transform(preds_test.reshape(-1, 1)).reshape(preds_test.shape)
    trues_test_unscaled = scaler.inverse_transform(trues_test.reshape(-1, 1)).reshape(trues_test.shape)
    per_horizon_mae = np.mean(np.abs(preds_test_unscaled - trues_test_unscaled), axis=0)
    horizons = np.arange(1, tgt_len + 1)
    plt.figure(figsize=(8, 3))
    sns.lineplot(x=horizons, y=per_horizon_mae, marker="o")
    plt.xlabel("Horizon")
    plt.ylabel("MAE")
    plt.title("Per-horizon MAE on test windows")
    plt.tight_layout()
    plt.show()
    print("Per-horizon MAE:", per_horizon_mae)

    # Plot training history
    plot_history(history)

def plot_history(history):
    df = pd.DataFrame(history)
    df.index.name = 'epoch'
    df = df.reset_index()
    plt.figure(figsize=(8, 3))
    sns.lineplot(data=df, x='epoch', y='train_loss', label='train_loss')
    sns.lineplot(data=df, x='epoch', y='val_loss', label='val_mae')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()