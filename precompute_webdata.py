"""
Precompute the web-platform dataset: mid + 6 features + P(up) + real label,
for every second. Output: web_data.json (downsampled) + web_data_full.csv.

Reuses the exact CNN-LSTM (cnn_lstm_v2.pt) and scaler so P(up) matches the
experiments. Real label = actual 30s-ahead direction (the ground truth).
"""
import json, numpy as np, pandas as pd, joblib, torch
from train_cnn_lstm_v2 import (CNNLSTM, add_directional_features,
                               make_binary_labels, FEATURES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 60

df = pd.read_csv("states_eth_20251201.csv").sort_values("ts").reset_index(drop=True)
df = add_directional_features(df)

# real 30s-ahead label (1 up, 0 down, -1 chop) — ground truth
real = make_binary_labels(df, horizon_s=30, move_mult=1.5)

# P(up) from the trained model, exactly as the experiments compute it
scaler = joblib.load("scaler_v2.joblib")
X = scaler.transform(df[FEATURES].to_numpy(np.float32)).astype(np.float32)
model = CNNLSTM(len(FEATURES)).to(DEVICE)
model.load_state_dict(torch.load("cnn_lstm_v2.pt", map_location=DEVICE))
model.eval()

pup = np.full(len(df), 0.5, np.float32)
idx = list(range(SEQ, len(df)))
with torch.no_grad():
    for s in range(0, len(idx), 4096):
        b = idx[s:s+4096]
        seqs = np.stack([X[i-SEQ:i] for i in b]).astype(np.float32)
        p = torch.softmax(model(torch.from_numpy(seqs).to(DEVICE)), 1)[:, 1]
        for k, i in enumerate(b):
            pup[i] = float(p[k])

out = pd.DataFrame({
    "t": np.arange(len(df)),
    "mid": df["mid"].round(2),
    "spread": df["spread"].round(4),
    "obi": df["obi"].round(4),
    "depth": df["depth"].round(1),
    "volatility": (df["volatility"]*1e4).round(3),   # scaled to readable units
    "intensity": df["trade_intensity"].round(2),
    "queue": df["queue_at_best"].round(2),
    "pup": np.round(pup, 4),
    "pred": (pup > 0.5).astype(int),                 # model call: 1 up, 0 down
    "has_pred": (np.arange(len(df)) >= SEQ).astype(int),  # 0 for first 60s (no window yet)
    "real": real,                                    # truth: 1/0/-1(chop)
})
out.to_csv("web_data_full.csv", index=False)

# downsample for the browser (every 10s -> ~8640 points, smooth + light)
ds = out.iloc[::10].reset_index(drop=True)
ds.to_json("web_data.json", orient="records")

# accuracy on decisive (non-chop) points where a window exists
mask = (out["real"] >= 0) & (out["t"] >= SEQ)
acc = (out.loc[mask, "pred"] == out.loc[mask, "real"]).mean()
print(f"Rows: {len(out):,} | downsampled: {len(ds):,}")
print(f"Directional accuracy (decisive pts): {acc*100:.1f}%")
print("Saved -> web_data.json, web_data_full.csv")