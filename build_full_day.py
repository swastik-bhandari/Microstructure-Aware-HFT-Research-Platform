import argparse
import time
import numpy as np
import pandas as pd

from read_data import read_orders, read_trades
from lob_engine import LOB, FeatureEngine


def build_full_day(date="2025-12-01", coin="eth", interval_ms=1000,
                   depth_levels=5, order_dir="order_statuses", map_dir="mapdir"):
    """Replay all 24 hours through one LOB; return (states_df, lob)."""
    lob = LOB()
    feat = FeatureEngine(depth_levels=depth_levels)
    interval_ns = interval_ms * 1_000_000

    states = []
    next_sample = None
    total_events = 0
    t_start = time.time()

    for hour in range(24):
        try:
            orders = read_orders(order_dir, map_dir, date=date,
                                 hour=hour, coin=coin)
        except ValueError:
            print(f"  hour {hour:02d}: no data, skipping")
            continue

        orders = orders.sort_values("ts").reset_index(drop=True)
        ts_ns = orders["ts"].astype("int64").to_numpy()

        oid_a    = orders["oid"].to_numpy()
        status_a = orders["status"].to_numpy()
        px_a     = orders["limitPx"].to_numpy()
        sz_a     = orders["sz"].to_numpy()
        isask_a  = orders["isAsk"].to_numpy()
        otype_a  = orders["orderType"].to_numpy()
        tif_a    = orders["tif"].to_numpy()

        if next_sample is None:
            next_sample = ts_ns[0] + interval_ns

        for i in range(len(orders)):
            t = ts_ns[i]
            while t >= next_sample:
                s = feat.sample(lob, next_sample)
                if s is not None:
                    s["hour"] = hour
                    states.append(s)
                next_sample += interval_ns

            lob.process_event(oid_a[i], status_a[i], px_a[i], sz_a[i],
                              isask_a[i], otype_a[i], tif_a[i], t)

        total_events += len(orders)
        bb, ba = lob.best_bid(), lob.best_ask()
        print(f"  hour {hour:02d}: {len(orders):>9,} events | "
              f"book {bb}/{ba} | {len(states):>6,} states so far")

    df = pd.DataFrame(states)
    elapsed = time.time() - t_start
    print(f"\nProcessed {total_events:,} events in {elapsed:.1f}s")
    print(f"Total rejected orders (full day): {lob.rejected_count:,}")
    return df, lob


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-12-01")
    ap.add_argument("--coin", default="eth")
    ap.add_argument("--interval-ms", type=int, default=1000)
    args = ap.parse_args()

    print(f"Building full-day states for {args.coin.upper()} {args.date} "
          f"@ {args.interval_ms}ms ...\n")

    states, lob = build_full_day(date=args.date, coin=args.coin,
                                 interval_ms=args.interval_ms)

    out = f"states_{args.coin}_{args.date.replace('-','')}.csv"
    states.to_csv(out, index=False)
    print(f"\nSaved {len(states):,} state vectors -> {out}")

    print("\n=== Feature sanity (full day) ===")
    print(f"spread > 0 always:   {(states['spread'] > 0).all()}")
    print(f"obi in [0,1] always: {states['obi'].between(0,1).all()}")
    print("\n=== Feature distributions (full day) ===")
    print(states[["spread", "obi", "depth", "volatility",
                  "trade_intensity", "queue_at_best"]].describe())

    # cross-check against full-day trades
    try:
        trades = read_trades("trades", date=args.date, coins=["ETH"])
        print(f"\nTrade price range (full day): "
              f"{trades['px'].min():.2f} - {trades['px'].max():.2f}")
        print(f"State mid range (full day):   "
              f"{states['mid'].min():.2f} - {states['mid'].max():.2f}")
    except ValueError:
        pass