"""
HFT Research Platform — Layer 3a v2: CNN-LSTM binary direction model
=====================================================================

Changes vs v1 (which underperformed baseline):
  - BINARY target (up vs down), 'flat'/chop excluded from training entirely
    -> cleaner, more learnable signal; matches what PPO needs (P(up))
  - SIGNED directional features added (v1 used only magnitude features):
      mid_return, mid_accel, obi_trend, signed_flow_proxy
  - Threshold defines a *meaningful* move; tiny chop is dropped, not forced
    into a class.

Output: P(up) probability -> 7th feature for the PPO agent.

Usage:
    python train_cnn_lstm_v2.py
    python train_cnn_lstm_v2.py --horizon 30 --seq-len 60 --move-mult 1.5
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_FEATURES = ["spread", "obi", "depth", "volatility",
                 "trade_intensity", "queue_at_best"]
# signed/directional features engineered below
DERIVED_FEATURES = ["mid_return", "mid_accel", "obi_trend", "signed_flow"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES


# ---------------------------------------------------------------------------
# Feature engineering: add SIGNED directional signals
# ---------------------------------------------------------------------------

def add_directional_features(df):
    df = df.copy()
    mid = df["mid"]

    # recent return (momentum) and its change (acceleration)
    df["mid_return"] = mid.pct_change().fillna(0.0)
    df["mid_accel"] = df["mid_return"].diff().fillna(0.0)

    # is order-book imbalance building up or unwinding? (signed)
    df["obi_trend"] = (df["obi"] - 0.5).rolling(10, min_periods=1).mean().fillna(0.0)

    # signed flow proxy: imbalance * intensity, centered
    # (high intensity with buy-skewed book -> positive; sell-skewed -> negative)
    df["signed_flow"] = ((df["obi"] - 0.5) * df["trade_intensity"]).fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Binary labeling with a meaningful-move threshold
# ---------------------------------------------------------------------------

def make_binary_labels(df, horizon_s=30, interval_s=1, move_mult=1.5):
    """Label up(1)/down(0); chop within +/-threshold -> -1 (excluded).

    threshold = move_mult * rolling vol of mid returns. Larger move_mult ->
    only decisive moves are labeled -> more learnable, fewer samples.
    """
    steps = horizon_s // interval_s
    mid = df["mid"].to_numpy()
    future = np.roll(mid, -steps)
    future[-steps:] = np.nan
    fwd_ret = (future - mid) / mid

    roll_vol = df["mid"].pct_change().rolling(60, min_periods=10).std().to_numpy()
    roll_vol = np.nan_to_num(roll_vol, nan=np.nanmedian(roll_vol))
    thr = move_mult * roll_vol

    labels = np.full(len(df), -1, dtype=np.int64)   # default: excluded chop
    labels[fwd_ret > thr] = 1     # up
    labels[fwd_ret < -thr] = 0    # down
    labels[np.isnan(fwd_ret)] = -1
    return labels


def make_sequences(X, y, seq_len):
    seqs, tgts = [], []
    for i in range(seq_len, len(X)):
        if y[i] < 0:
            continue
        seqs.append(X[i - seq_len:i])
        tgts.append(y[i])
    return (np.asarray(seqs, dtype=np.float32),
            np.asarray(tgts, dtype=np.int64))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNNLSTM(nn.Module):
    def __init__(self, n_features, cnn_ch=48, lstm_hidden=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, cnn_ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch, 3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_ch, lstm_hidden, batch_first=True, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 48), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(48, 2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    pc_c = np.zeros(2); pc_t = np.zeros(2)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item(); total += len(yb)
            for c in range(2):
                m = yb == c
                pc_t[c] += m.sum().item()
                pc_c[c] += (pred[m] == c).sum().item()
    acc = correct / max(total, 1)
    pca = np.divide(pc_c, pc_t, out=np.zeros(2), where=pc_t > 0)
    return acc, pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="states_eth_20251201.csv")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--move-mult", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    df = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    df = add_directional_features(df)
    print(f"Loaded {len(df):,} states; features = {len(FEATURES)}")

    labels = make_binary_labels(df, args.horizon, move_mult=args.move_mult)
    valid = labels >= 0
    c = np.bincount(labels[valid], minlength=2)
    print(f"\nBinary labels (move_mult={args.move_mult}, horizon={args.horizon}s):")
    print(f"  down: {c[0]:,} ({100*c[0]/valid.sum():.1f}%)")
    print(f"  up:   {c[1]:,} ({100*c[1]/valid.sum():.1f}%)")
    print(f"  excluded chop: {(~valid).sum() - args.horizon:,}")
    maj = c.max() / valid.sum()
    print(f"  majority baseline: {100*maj:.1f}%")

    n = len(df)
    tr_end, va_end = int(n*0.70), int(n*0.85)
    X_raw = df[FEATURES].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(X_raw[:tr_end])
    X = scaler.transform(X_raw).astype(np.float32)
    joblib.dump(scaler, "scaler_v2.joblib")

    Xtr, ytr = make_sequences(X[:tr_end], labels[:tr_end], args.seq_len)
    Xva, yva = make_sequences(X[tr_end:va_end], labels[tr_end:va_end], args.seq_len)
    Xte, yte = make_sequences(X[va_end:], labels[va_end:], args.seq_len)
    print(f"\nSequences  train={len(Xtr):,}  val={len(Xva):,}  test={len(Xte):,}")
    if len(Xtr) < 500:
        print("WARNING: very few training sequences — lower --move-mult.")

    def loader(Xa, ya, sh):
        return DataLoader(TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
                          batch_size=args.batch, shuffle=sh)
    tl, vl, tel = loader(Xtr, ytr, True), loader(Xva, yva, False), loader(Xte, yte, False)

    cc = np.bincount(ytr, minlength=2)
    w = torch.tensor(cc.sum() / (2*np.maximum(cc, 1)), dtype=torch.float32).to(DEVICE)
    model = CNNLSTM(len(FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    print(f"\nTraining ({sum(p.numel() for p in model.parameters()):,} params)...")
    best = 0.0; patience = 0
    for ep in range(1, args.epochs+1):
        model.train(); tot = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item()*len(yb)
        va, vpc = evaluate(model, vl)
        flag = ""
        if va > best:
            best = va; patience = 0
            torch.save(model.state_dict(), "cnn_lstm_v2.pt"); flag = "  <- saved"
        else:
            patience += 1
        print(f"  epoch {ep:2d}  loss {tot/max(len(Xtr),1):.4f}  "
              f"val_acc {va:.3f}  [down {vpc[0]:.2f} up {vpc[1]:.2f}]{flag}")
        if patience >= 6:
            print("  early stop (no val improvement in 6 epochs)")
            break

    model.load_state_dict(torch.load("cnn_lstm_v2.pt"))
    ta, tpc = evaluate(model, tel)
    print(f"\n=== TEST RESULTS (binary) ===")
    print(f"  accuracy:          {ta:.3f}")
    print(f"  majority baseline: {maj:.3f}")
    print(f"  random:            0.500")
    print(f"  per-class:  down {tpc[0]:.2f}  up {tpc[1]:.2f}")
    print(f"  edge over majority: {100*(ta-maj):+.1f} pts")
    print(f"  edge over random:   {100*(ta-0.5):+.1f} pts")

    json.dump({"horizon_s": args.horizon, "seq_len": args.seq_len,
               "move_mult": args.move_mult, "features": FEATURES,
               "test_acc": float(ta), "majority": float(maj)},
              open("cnn_lstm_v2_meta.json", "w"), indent=2)
    print("\nSaved -> cnn_lstm_v2.pt, scaler_v2.joblib, cnn_lstm_v2_meta.json")


if __name__ == "__main__":
    main()