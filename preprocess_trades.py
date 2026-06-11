"""
One-time preprocessing: convert the day's ETH trades into a compact,
time-sorted "trade tape" (.npz) used by the execution environment for
REAL fill attribution.

For each trade we keep:
    ts_ns : int64   nanosecond timestamp
    px    : float32 execution price
    sz    : float32 execution size
    isB   : bool    True if buy-aggressor ("B") — the trades that can fill
                    a resting SELL order

Usage:
    python preprocess_trades.py
    python preprocess_trades.py --date 2025-12-01 --coin ETH
"""

import argparse
import numpy as np
import pandas as pd
from read_data import read_trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-12-01")
    ap.add_argument("--coin", default="ETH")
    ap.add_argument("--trades-dir", default="trades")
    args = ap.parse_args()

    frames = []
    for hour in range(24):
        try:
            df = read_trades(args.trades_dir, date=args.date, hour=hour,
                             coins=[args.coin])
        except (ValueError, FileNotFoundError):
            print(f"  hour {hour:02d}: missing, skipped")
            continue
        frames.append(df)
        print(f"  hour {hour:02d}: {len(df):,} {args.coin} trades")

    trades = pd.concat(frames, ignore_index=True)

    ts_ns = pd.to_datetime(trades["time"]).astype("int64").to_numpy()
    px = trades["px"].astype(np.float32).to_numpy()
    sz = trades["sz"].astype(np.float32).to_numpy()
    isB = (trades["side"] == "B").to_numpy()

    order = np.argsort(ts_ns, kind="stable")
    ts_ns, px, sz, isB = ts_ns[order], px[order], sz[order], isB[order]

    out = f"trade_tape_{args.coin.lower()}_{args.date.replace('-','')}.npz"
    np.savez_compressed(out, ts_ns=ts_ns, px=px, sz=sz, isB=isB)

    print(f"\nSaved {len(ts_ns):,} trades -> {out}")
    print(f"  buy-aggressor (can fill our sells): {isB.sum():,} "
          f"({100*isB.mean():.1f}%)")
    print(f"  px range: {px.min():.2f} - {px.max():.2f}")


if __name__ == "__main__":
    main()
