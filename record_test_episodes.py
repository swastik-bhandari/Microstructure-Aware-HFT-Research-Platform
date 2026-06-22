"""
record_test_episodes.py
=======================
Record the trained PPO agent's PER-STEP behaviour on the TEST split only.

This is the honest analogue of run_variance.py's eval_policy(): it uses the SAME
70/30 chronological split, samples random 60s episodes from the held-out test
region, and runs the trained agent deterministically -- but unlike eval_policy
(which keeps only the final shortfall) it records every action, fill, inventory
level, P(up) and running shortfall.

No hand-picked windows. No regime labels. Just real out-of-sample episodes,
sampled uniformly at random from test data, exactly like evaluation does.

Usage:
    python record_test_episodes.py --model models/ppo7_seed1000.zip --n 8 --seed 7
    python record_test_episodes.py --model models/ppo7_sizeaware_seed1000.zip --n 8

Output: test_episodes.json  (list of episodes, each a list of per-step dicts)
"""
import argparse, json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from execution_env import ExecutionEnv, TradeTape
from execution_env_v2 import ExecutionEnvV2
from run_variance import compute_signal

# 3-action env (ppo6 / ppo7): hold / limit / market
ACTNAME_V1 = {0: "HOLD", 1: "LIMIT", 2: "MARKET"}
# 7-action size-aware env (ppo*_sizeaware): hold + {limit,market} x {small,medium,large}
ACTNAME_V2 = {0: "HOLD",
              1: "LIMIT", 2: "LIMIT", 3: "LIMIT",
              4: "MARKET", 5: "MARKET", 6: "MARKET"}
SIZE_V2 = {0: "", 1: "small", 2: "medium", 3: "large",
           4: "small", 5: "medium", 6: "large"}


def record_episode(model, env, actname, split_offset, sizemap=None):
    """Run ONE deterministic episode on the given env; return per-step rows."""
    obs, _ = env.reset()                       # random start within env's data
    rows, done = [], False
    info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        inv_before = env.inventory
        t_local = env.t                         # index into the TEST slice df
        t_abs = split_offset + t_local          # absolute row in the full day
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        sold = inv_before - env.inventory
        avg = (env.realized_value / env.executed_qty) if env.executed_qty > 1e-9 else env.arrival_mid
        sf = (env.arrival_mid - avg) / env.arrival_mid * 1e4 if env.executed_qty > 1e-9 else 0.0
        # real wall-clock time-of-day from the nanosecond ts column
        sec = int(env.df["ts"].iloc[t_local] // 1_000_000_000 % 86400) if "ts" in env.df else t_abs
        hms = f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}"
        rows.append({
            "t": hms,                           # human clock time, e.g. "20:11:55"
            "sec": sec,                         # seconds-since-midnight UTC
            "idx": int(t_abs),                  # ABSOLUTE row in the full day
            "mid": round(float(env.mid[t_local]), 2),
            "action": actname.get(a, f"A{a}"),
            "size": (sizemap.get(a, "") if sizemap else ""),
            "sold": round(float(sold), 3),
            "inv": round(float(env.inventory), 3),
            "pup": round(float(env.signal.iloc[t_local]), 3) if env.use_signal else None,
            "shortfall": round(float(sf), 2),
        })
    return {
        "start_idx": int(split_offset + env.start),   # ABSOLUTE row in the full day
        "start_local": int(env.start),                # index within the test slice
        "arrival": round(float(env.arrival_mid), 2),
        "final_shortfall": round(float(info.get("shortfall_bps", sf)), 2),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="states_eth_20251201.csv")
    ap.add_argument("--tape", default="trade_tape_eth_20251201.npz")
    ap.add_argument("--model", default="models/ppo7_seed1000.zip")
    ap.add_argument("--out", default="test_episodes.json")
    ap.add_argument("--n", type=int, default=8, help="number of test episodes to record")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed for episode starts")
    ap.add_argument("--split", type=float, default=0.70)
    ap.add_argument("--use-signal", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    signal = compute_signal(df)                  # frozen CNN-LSTM P(up), same as training
    tape = TradeTape(args.tape)

    # ---- SAME 70/30 chronological split as run_variance.py ----
    n = len(df)
    split = int(n * args.split)
    df_te = df.iloc[split:].reset_index(drop=True)
    sig_te = signal.iloc[split:].reset_index(drop=True)
    print(f"total states={n} | split at {split} | TEST region = rows {split}..{n-1} "
          f"({n - split} states, ~{(n - split) / 3600:.1f}h)")

    # ---- pick env + action map based on the model ----
    is_sizeaware = "sizeaware" in args.model.lower()
    use_sig = ("ppo7" in args.model) or is_sizeaware   # ppo7 and size-aware use P(up)

    if is_sizeaware:
        env = ExecutionEnvV2(df_te, sig_te, use_signal=use_sig, trade_tape=tape,
                             target_qty=10.0, horizon_steps=60, seed=args.seed)
        actname, sizemap = ACTNAME_V2, SIZE_V2
        print(f"size-aware model -> ExecutionEnvV2 (7 actions)")
    else:
        env = ExecutionEnv(df_te, sig_te, use_signal=use_sig, trade_tape=tape,
                           target_qty=10.0, horizon_steps=60, seed=args.seed)
        actname, sizemap = ACTNAME_V1, None
        print(f"standard model -> ExecutionEnv (3 actions)")

    model = PPO.load(args.model, device="cpu")
    print(f"loaded {args.model} | use_signal={use_sig} | recording {args.n} random test episodes")

    episodes = [record_episode(model, env, actname, split, sizemap) for _ in range(args.n)]

    # quick honest summary
    from collections import Counter
    print("\n# per-episode summary (random test windows):")
    for i, e in enumerate(episodes):
        acts = Counter(r["action"] for r in e["rows"])
        lim_sold = sum(r["sold"] for r in e["rows"] if r["action"] == "LIMIT")
        mkt_sold = sum(r["sold"] for r in e["rows"] if r["action"] == "MARKET")
        print(f"  ep{i}: start_idx={e['start_idx']:>5} | {dict(acts)} "
              f"| LIMIT-filled {lim_sold:.2f} / MARKET-filled {mkt_sold:.2f} ETH "
              f"| final {e['final_shortfall']:+.2f} bps")

    allsf = [e["final_shortfall"] for e in episodes]
    print(f"\n# mean final shortfall over {len(episodes)} test episodes: "
          f"{np.mean(allsf):.2f} bps (std {np.std(allsf):.2f})")

    json.dump(episodes, open(args.out, "w"), separators=(",", ":"))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()