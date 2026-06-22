"""
inspect_raw.py — print metadata of the RAW Hyperliquid data (orders + trades).

Run this from the folder that contains order_statuses/, trades/, and mapdir/.
It uses the official read_data.py reader, so the numbers are exactly what the
pipeline sees. Inspects ONE hour by default (fast); pass --hour or --all-hours.

    python inspect_raw.py
    python inspect_raw.py --date 2025-12-01 --hour 12
"""
import argparse, os, glob
import numpy as np
import pandas as pd
from read_data import read_orders, read_trades


def hr(t): print("\n" + "=" * 70 + f"\n {t}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-12-01")
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--coin", default="eth")
    ap.add_argument("--order-dir", default="order_statuses")
    ap.add_argument("--trades-dir", default="trades")
    ap.add_argument("--map-dir", default="mapdir")
    args = ap.parse_args()

    # ---- physical files on disk -------------------------------------------
    hr("RAW FILES ON DISK")
    d = args.date.replace("-", "")
    for label, pat in [
        ("order_statuses", f"{args.order_dir}/{d}/*.data.gz"),
        ("trades",         f"{args.trades_dir}/{d}/*.gz"),
        ("mapdir",         f"{args.map_dir}/*.csv"),
    ]:
        files = sorted(glob.glob(pat))
        total = sum(os.path.getsize(f) for f in files)
        print(f"  {label:15} {len(files)} files, {total/1e6:.1f} MB total")
        for f in files[:3]:
            print(f"      {f}  ({os.path.getsize(f)/1e6:.2f} MB)")
        if len(files) > 3:
            print(f"      ... (+{len(files)-3} more)")

    # ---- ORDER STATUSES ----------------------------------------------------
    hr(f"ORDER STATUSES — {args.coin} {args.date} hour {args.hour}")
    o = read_orders(args.order_dir, args.map_dir,
                    date=args.date, hour=args.hour, coin=args.coin)
    print(f"  rows (events): {len(o):,}")
    print(f"  columns/dtypes: {dict(o.dtypes.astype(str))}")
    print(f"  ts range: {o['ts'].min()} .. {o['ts'].max()}")
    if "statusId" in o or "status" in o:
        scol = "status" if "status" in o else "statusId"
        print(f"  status breakdown:\n{o[scol].value_counts().to_string()}")
    if "tif" in o:
        print(f"  tif breakdown:\n{o['tif'].value_counts().to_string()}")
    if "isAsk" in o:
        print(f"  isAsk (true=sell): {dict(o['isAsk'].value_counts())}")
    pxcol = "limitPx" if "limitPx" in o else None
    if pxcol:
        print(f"  limitPx range: {o[pxcol].min():.2f} .. {o[pxcol].max():.2f}")
    print(f"  unique oids: {o['oid'].nunique():,}")
    print("  first 3 rows:")
    print(o.head(3).to_string())

    # ---- TRADES ------------------------------------------------------------
    hr(f"TRADES — {args.coin.upper()} {args.date} hour {args.hour}")
    t = read_trades(args.trades_dir, date=args.date, hour=args.hour,
                    coins=[args.coin.upper()])
    print(f"  rows (trades): {len(t):,}")
    print(f"  columns/dtypes: {dict(t.dtypes.astype(str))}")
    if "side" in t:
        print(f"  side breakdown (A=sell-agg, B=buy-agg): "
              f"{dict(t['side'].value_counts())}")
    if "px" in t:
        print(f"  px range: {t['px'].min():.2f} .. {t['px'].max():.2f}")
    if "sz" in t:
        print(f"  sz range: {t['sz'].min():.6f} .. {t['sz'].max():.4f}")
    print("  first 3 rows:")
    print(t.head(3).to_string())

    # ---- MAPDIR ------------------------------------------------------------
    hr("MAPDIR LOOKUP TABLES")
    for name in ["order_types.csv", "statuses.csv", "tifs.csv"]:
        p = os.path.join(args.map_dir, name)
        if os.path.exists(p):
            print(f"  {name}:")
            print("   ", open(p).read().strip().replace("\n", " | "))
    for name in ["users.csv", "value_stats.csv"]:
        p = os.path.join(args.map_dir, name)
        if os.path.exists(p):
            n = sum(1 for _ in open(p)) - 1
            print(f"  {name}: {n:,} rows ({os.path.getsize(p)/1e6:.1f} MB)")

    print("\nDone. This is the raw data exactly as the pipeline reads it.")


if __name__ == "__main__":
    main()