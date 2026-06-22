"""
inspect_data.py — print the real schema/metadata of every project artifact.

Run this to VERIFY (not trust) what each file contains: shapes, dtypes, columns,
ranges, and key stats. Reads only what exists; skips missing files gracefully.

    python inspect_data.py
"""
import os, json
import numpy as np

def hr(t): print("\n" + "=" * 70 + f"\n {t}\n" + "=" * 70)

def show_csv(path):
    import pandas as pd
    if not os.path.exists(path): print(f"  [missing] {path}"); return
    df = pd.read_csv(path)
    print(f"  {path}")
    print(f"    shape: {df.shape}")
    print(f"    columns/dtypes: {dict(df.dtypes.astype(str))}")
    if "ts" in df.columns:
        print(f"    ts range: {df['ts'].min()} .. {df['ts'].max()}")
    print(f"    head:\n{df.head(3).to_string().rstrip()}")

def show_json(path):
    if not os.path.exists(path): print(f"  [missing] {path}"); return
    print(f"  {path}")
    print("   ", json.dumps(json.load(open(path)), indent=2)[:1200])

def show_npz(path):
    if not os.path.exists(path): print(f"  [missing] {path}"); return
    z = np.load(path)
    print(f"  {path}")
    for k in z.files:
        a = z[k]
        print(f"    {k:8} shape={a.shape} dtype={a.dtype} sample={a[:3]}")
    if "isB" in z.files:
        print(f"    buy-aggressor frac: {float(z['isB'].mean()):.4f}")
    if "px" in z.files:
        print(f"    px range: {z['px'].min():.2f} .. {z['px'].max():.2f}")

hr("STATE TABLE (Layer 2 output)")
show_csv("states_eth_20251201.csv")

hr("TRADE TAPE (real-fill input)")
show_npz("trade_tape_eth_20251201.npz")

hr("CNN-LSTM METADATA (Layer 3a)")
show_json("cnn_lstm_v2_meta.json")

hr("RESULTS — IDEALIZED vs REAL FILLS")
show_json("variance_summary_IDEALIZED.json")
show_json("variance_summary_REALFILLS.json")

hr("SENSITIVITY SWEEP")
show_csv("sensitivity_results.csv")

hr("OUTLIER RE-TEST")
show_json("retest_alpha0.5_eta0.5_summary.json")

hr("FAIRNESS CHECK (size-aware actions)")
show_json("fairness_summary.json")

hr("TRAINED MODELS")
mdir = "models"
if os.path.isdir(mdir):
    for f in sorted(os.listdir(mdir)):
        print(f"  {mdir}/{f}  ({os.path.getsize(os.path.join(mdir,f))} bytes)")
else:
    print("  [missing] models/")

print("\nDone. Every number in PROJECT_README.md / DATA_FLOW.md traces to the above.")
