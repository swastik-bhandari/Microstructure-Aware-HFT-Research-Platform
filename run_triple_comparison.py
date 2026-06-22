"""
run_triple_comparison.py
========================
Runs PPO_static, PPO_sizeaware, and VWAP on the SAME N test windows.

All three use:
  - same 70/30 chronological split as run_variance.py
  - same TradeTape real fills
  - same random start points (fixed seed)

PPO_static   -> ExecutionEnv   (3 actions: hold/limit/market, fixed 0.25 ETH/step)
PPO_sizeaware -> ExecutionEnvV2 (7 actions: hold + {limit,market} x {small,med,large})
VWAP         -> ExecutionEnv   (qty_override bypass, env-agnostic fill result)

Outputs:
  triple_comparison.csv   -- per-episode row: start_idx, all 3 bps, pairwise gaps
  triple_summary.json     -- mean / median / std / win-rates for each agent
  triple_episodes.json    -- per-step recorded actions for 6 showcase episodes each

Usage:
    python run_triple_comparison.py --n 50 --seed 1
    python run_triple_comparison.py --n 50 --seed 1 --static models/ppo7_seed1000.zip
"""
import argparse, json, csv
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from execution_env import ExecutionEnv, TradeTape
from execution_env_v2 import ExecutionEnvV2
from run_variance import compute_signal

# ── helpers ──────────────────────────────────────────────────────────────────

def reset_to(env, start):
    """Pin the env to a fixed start without using the env's internal rng."""
    env.reset()
    env.start   = start
    env.t       = start
    env.steps_left    = env.H
    env.inventory     = env.target_qty
    env.arrival_mid   = env.mid[start]
    env.realized_value = 0.0
    env.executed_qty  = 0.0
    return env._obs()


def run_ppo(model, env, start):
    """Run a PPO model deterministically from a fixed start; return shortfall_bps + step log."""
    obs = reset_to(env, start)
    rows, done, info = [], False, {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        inv_before = env.inventory
        t_local = env.t
        sec = int(env.df["ts"].iloc[t_local] // 1_000_000_000 % 86400) if "ts" in env.df else t_local
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        sold = inv_before - env.inventory
        avg = (env.realized_value / env.executed_qty) if env.executed_qty > 1e-9 else env.arrival_mid
        sf  = (env.arrival_mid - avg) / env.arrival_mid * 1e4 if env.executed_qty > 1e-9 else 0.0
        rows.append({
            "t":   f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}",
            "sec": sec,
            "idx": int(env._split_offset + t_local) if hasattr(env, "_split_offset") else t_local,
            "mid": round(float(env.mid[t_local]), 2),
            "action": ["HOLD","LIMIT","MARKET","LIMIT","MARKET"][min(a,4)] if a < 5
                      else "MARKET",   # v2 large actions -> MARKET
            "sold": round(float(sold), 3),
            "inv":  round(float(env.inventory), 3),
            "pup":  round(float(env.signal.iloc[t_local]), 3) if env.use_signal else None,
            "shortfall": round(float(sf), 2),
        })
    return float(info.get("shortfall_bps", sf)), rows


def run_vwap(env, start):
    """Run VWAP from a fixed start; return shortfall_bps + step log."""
    obs = reset_to(env, start)
    H   = env.H
    vol = env.intensity[start:start + H].astype(np.float64)
    vol = np.where(vol > 0, vol, max(vol.mean(), 1e-9))
    weights = vol / vol.sum()
    rows, done, info, step_i = [], False, {}, 0
    while not done:
        target_sold = env.target_qty * weights[min(step_i, H - 1)]
        inv_before  = env.inventory
        t_local     = env.t
        sec = int(env.df["ts"].iloc[t_local] // 1_000_000_000 % 86400) if "ts" in env.df else t_local
        obs, r, term, trunc, info = env.step(2, qty_override=target_sold)
        done = term or trunc
        sold = inv_before - env.inventory
        avg  = (env.realized_value / env.executed_qty) if env.executed_qty > 1e-9 else env.arrival_mid
        sf   = (env.arrival_mid - avg) / env.arrival_mid * 1e4 if env.executed_qty > 1e-9 else 0.0
        rows.append({
            "t":   f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}",
            "sec": sec,
            "idx": t_local,
            "mid": round(float(env.mid[t_local]), 2),
            "action": "MARKET" if sold > 1e-9 else "HOLD",
            "sold": round(float(sold), 3),
            "inv":  round(float(env.inventory), 3),
            "pup":  None,
            "shortfall": round(float(sf), 2),
        })
        step_i += 1
    return float(info.get("shortfall_bps", sf)), rows


def stats(arr):
    a = np.array(arr, dtype=float)
    return {"mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "std":  round(float(a.std()), 3),
            "min":  round(float(a.min()), 3),
            "max":  round(float(a.max()), 3)}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",    default="states_eth_20251201.csv")
    ap.add_argument("--tape",    default="trade_tape_eth_20251201.npz")
    ap.add_argument("--static",  default="models/ppo7_seed1000.zip",       help="fixed-size PPO model")
    ap.add_argument("--sizeaware", default="models/ppo7_sizeaware_seed1000.zip")
    ap.add_argument("--n",       type=int, default=50)
    ap.add_argument("--seed",    type=int, default=1)
    ap.add_argument("--split",   type=float, default=0.70)
    ap.add_argument("--showcase-n", type=int, default=6, help="episodes per agent in showcase JSON")
    ap.add_argument("--out-csv",  default="triple_comparison.csv")
    ap.add_argument("--out-json", default="triple_summary.json")
    ap.add_argument("--out-ep",   default="triple_episodes.json")
    args = ap.parse_args()

    # ── data ────────────────────────────────────────────────────────────────
    df     = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    signal = compute_signal(df)
    tape   = TradeTape(args.tape)
    n      = len(df)
    split  = int(n * args.split)
    df_te  = df.iloc[split:].reset_index(drop=True)
    sig_te = signal.iloc[split:].reset_index(drop=True)
    print(f"states={n} | split={split} | test={n-split} ({(n-split)/3600:.1f}h)\n")

    # ── envs ────────────────────────────────────────────────────────────────
    env_static = ExecutionEnv(df_te, sig_te, use_signal=True,  trade_tape=tape,
                              target_qty=10.0, horizon_steps=60, seed=args.seed)
    env_sa     = ExecutionEnvV2(df_te, sig_te, use_signal=True, trade_tape=tape,
                                target_qty=10.0, horizon_steps=60, seed=args.seed)
    env_vwap   = ExecutionEnv(df_te, sig_te, use_signal=False, trade_tape=tape,
                              target_qty=10.0, horizon_steps=60, seed=args.seed)

    # attach split offset so rows can report absolute idx
    for env in (env_static, env_sa, env_vwap):
        env._split_offset = split

    # ── models ──────────────────────────────────────────────────────────────
    m_static = PPO.load(args.static,   device="cpu")
    m_sa     = PPO.load(args.sizeaware, device="cpu")
    print(f"static : {args.static}")
    print(f"size-aware: {args.sizeaware}")
    print(f"running {args.n} paired windows (seed={args.seed})\n")

    # ── sample fixed start points ────────────────────────────────────────────
    rng       = np.random.default_rng(args.seed)
    max_start = len(df_te) - 60 - 1
    starts    = [int(rng.integers(0, max_start)) for _ in range(args.n)]

    # ── run all three on every window ────────────────────────────────────────
    rows_csv   = []
    ep_static  = []
    ep_sa      = []
    ep_vwap    = []

    print(f"{'ep':>3} {'start_abs':>9} {'STATIC':>9} {'SIZEAWARE':>9} {'VWAP':>9} "
          f"{'S-V':>8} {'SA-V':>8}")
    print("-" * 64)

    for k, start in enumerate(starts):
        sf_st,  rows_st  = run_ppo( m_static, env_static, start)
        sf_sa,  rows_sa  = run_ppo( m_sa,     env_sa,     start)
        sf_vw,  rows_vw  = run_vwap(env_vwap, start)

        abs_idx = split + start
        rows_csv.append({
            "episode":          k,
            "start_local":      start,
            "start_idx_abs":    abs_idx,
            "static_bps":       round(sf_st, 2),
            "sizeaware_bps":    round(sf_sa, 2),
            "vwap_bps":         round(sf_vw, 2),
            "static_vs_vwap":   round(sf_st - sf_vw, 2),
            "sizeaware_vs_vwap": round(sf_sa - sf_vw, 2),
        })

        ep_static.append({"start_idx": abs_idx, "start_local": start,
                          "arrival": round(float(env_static.arrival_mid), 2),
                          "final_shortfall": round(sf_st, 2), "rows": rows_st})
        ep_sa.append(    {"start_idx": abs_idx, "start_local": start,
                          "arrival": round(float(env_sa.arrival_mid), 2),
                          "final_shortfall": round(sf_sa, 2), "rows": rows_sa})
        ep_vwap.append(  {"start_idx": abs_idx, "start_local": start,
                          "arrival": round(float(env_vwap.arrival_mid), 2),
                          "final_shortfall": round(sf_vw, 2), "rows": rows_vw})

        sv  = sf_st - sf_vw
        sav = sf_sa - sf_vw
        print(f"{k:>3} {abs_idx:>9} {sf_st:>9.2f} {sf_sa:>9.2f} {sf_vw:>9.2f} "
              f"{sv:>+8.2f} {sav:>+8.2f}")

    # ── summary ─────────────────────────────────────────────────────────────
    s_bps  = [r["static_bps"]   for r in rows_csv]
    sa_bps = [r["sizeaware_bps"] for r in rows_csv]
    v_bps  = [r["vwap_bps"]     for r in rows_csv]
    sv_gap = [r["static_vs_vwap"] for r in rows_csv]
    sav_gap= [r["sizeaware_vs_vwap"] for r in rows_csv]

    static_wins_vwap    = sum(1 for x in sv_gap  if x < 0)
    sizeaware_wins_vwap = sum(1 for x in sav_gap if x < 0)
    sizeaware_wins_static = sum(1 for i in range(args.n) if sa_bps[i] < s_bps[i])

    print("-" * 64)
    print(f"{'mean':>3} {'':>9} {np.mean(s_bps):>9.2f} {np.mean(sa_bps):>9.2f} "
          f"{np.mean(v_bps):>9.2f} {np.mean(sv_gap):>+8.2f} {np.mean(sav_gap):>+8.2f}")
    print(f"{'med':>3} {'':>9} {np.median(s_bps):>9.2f} {np.median(sa_bps):>9.2f} "
          f"{np.median(v_bps):>9.2f} {np.median(sv_gap):>+8.2f} {np.median(sav_gap):>+8.2f}")
    print(f"{'std':>3} {'':>9} {np.std(s_bps):>9.2f} {np.std(sa_bps):>9.2f} "
          f"{np.std(v_bps):>9.2f} {np.std(sv_gap):>8.2f} {np.std(sav_gap):>8.2f}")
    print()
    print(f"static   beats VWAP:    {static_wins_vwap}/{args.n}")
    print(f"sizeaware beats VWAP:   {sizeaware_wins_vwap}/{args.n}")
    print(f"sizeaware beats static: {sizeaware_wins_static}/{args.n}")

    # ── write CSV ────────────────────────────────────────────────────────────
    fieldnames = ["episode","start_local","start_idx_abs",
                  "static_bps","sizeaware_bps","vwap_bps",
                  "static_vs_vwap","sizeaware_vs_vwap"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows_csv)
    print(f"\nsaved -> {args.out_csv}")

    # ── write summary JSON ────────────────────────────────────────────────────
    summary = {
        "n": args.n, "seed": args.seed,
        "models": {"static": args.static, "sizeaware": args.sizeaware},
        "static":    {**stats(s_bps),
                      "beats_vwap": static_wins_vwap,
                      "beats_vwap_pct": round(100*static_wins_vwap/args.n,1)},
        "sizeaware": {**stats(sa_bps),
                      "beats_vwap": sizeaware_wins_vwap,
                      "beats_vwap_pct": round(100*sizeaware_wins_vwap/args.n,1),
                      "beats_static": sizeaware_wins_static,
                      "beats_static_pct": round(100*sizeaware_wins_static/args.n,1)},
        "vwap":      stats(v_bps),
        "gaps": {
            "static_vs_vwap":    stats(sv_gap),
            "sizeaware_vs_vwap": stats(sav_gap),
        }
    }
    json.dump(summary, open(args.out_json,"w"), indent=2)
    print(f"saved -> {args.out_json}")

    # ── write showcase episodes (first showcase_n of each) ───────────────────
    sn = args.showcase_n
    showcase = {
        "static":    ep_static[:sn],
        "sizeaware": ep_sa[:sn],
        "vwap":      ep_vwap[:sn],
    }
    json.dump(showcase, open(args.out_ep,"w"), separators=(",",":"))
    print(f"saved -> {args.out_ep}  ({sn} episodes per agent)")


if __name__ == "__main__":
    main()
