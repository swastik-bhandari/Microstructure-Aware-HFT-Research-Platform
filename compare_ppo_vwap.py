"""
compare_ppo_vwap.py
===================
Paired PPO vs VWAP comparison on the SAME test windows.

For each of N random test-set episodes we fix the start point, then run BOTH
PPO and VWAP on that identical window (same data, same fills, same config) and
record each one's implementation shortfall. This is a PAIRED comparison: it
controls for which windows we sampled, so the per-episode gap is meaningful even
when the means are noisy.

Mirrors run_variance.py exactly:
  - same 70/30 chronological split
  - same ExecutionEnv + TradeTape (real fills)
  - same run_vwap (volume-weighted market slices)
  - PPO deterministic, 7-feature (loads models/ppo7_seed1000.zip by default)

Usage:
    python compare_ppo_vwap.py --model models/ppo7_seed1000.zip --n 12 --seed 1

Output: prints a paired table + saves compare_ppo_vwap.json
"""
import argparse, json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from execution_env import ExecutionEnv, TradeTape, run_vwap
from execution_env_v2 import ExecutionEnvV2
from run_variance import compute_signal


def run_ppo_fixed(model, env, start):
    """Run PPO deterministically from a FIXED start; return shortfall_bps."""
    env.reset()
    # pin the window (mirror reset()'s own assignments)
    env.start = start; env.t = start
    env.steps_left = env.H; env.inventory = env.target_qty
    env.arrival_mid = env.mid[start]; env.realized_value = 0.0; env.executed_qty = 0.0
    obs = env._obs()
    done, info = False, {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(a))
        done = term or trunc
    return float(info.get("shortfall_bps", np.nan))


def run_vwap_fixed(env, start):
    """Run VWAP from a FIXED start; return shortfall_bps."""
    env.reset()
    env.start = start; env.t = start
    env.steps_left = env.H; env.inventory = env.target_qty
    env.arrival_mid = env.mid[start]; env.realized_value = 0.0; env.executed_qty = 0.0
    # replicate run_vwap's schedule, but on the pinned start
    H = env.H
    vol = env.intensity[start:start + H].astype(np.float64)
    vol = np.where(vol > 0, vol, vol.mean() if vol.mean() > 0 else 1.0)
    weights = vol / vol.sum()
    done, info, step_i = False, {}, 0
    while not done:
        slice_qty = env.target_qty * weights[min(step_i, H - 1)]
        obs, r, term, trunc, info = env.step(2, qty_override=slice_qty)
        step_i += 1
        done = term or trunc
    return float(info.get("shortfall_bps", np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="states_eth_20251201.csv")
    ap.add_argument("--tape", default="trade_tape_eth_20251201.npz")
    ap.add_argument("--model", default="models/ppo7_seed1000.zip")
    ap.add_argument("--out", default="compare_ppo_vwap.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", type=float, default=0.70)
    args = ap.parse_args()

    df = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    signal = compute_signal(df)
    tape = TradeTape(args.tape)

    n = len(df)
    split = int(n * args.split)
    df_te = df.iloc[split:].reset_index(drop=True)
    sig_te = signal.iloc[split:].reset_index(drop=True)
    print(f"total states={n} | split at {split} | TEST = rows {split}..{n-1} "
          f"({n - split} states, ~{(n - split)/3600:.1f}h)\n")

    is_sizeaware = "sizeaware" in args.model.lower()
    use_sig = ("ppo7" in args.model) or is_sizeaware

    # IMPORTANT: PPO and VWAP must run in the SAME env type, because the env
    # defines the fill model. Size-aware models need ExecutionEnvV2 (7 actions);
    # standard ppo6/ppo7 need ExecutionEnv (3 actions). run_fairness.py runs BOTH
    # the size-aware agent and VWAP inside ExecutionEnvV2 -- we mirror that.
    EnvCls = ExecutionEnvV2 if is_sizeaware else ExecutionEnv
    print(f"using {'ExecutionEnvV2 (7-action)' if is_sizeaware else 'ExecutionEnv (3-action)'} "
          f"for both PPO and VWAP\n")

    env_ppo = EnvCls(df_te, sig_te, use_signal=use_sig, trade_tape=tape,
                     target_qty=10.0, horizon_steps=60, seed=args.seed)
    env_vwap = EnvCls(df_te, sig_te, use_signal=use_sig, trade_tape=tape,
                      target_qty=10.0, horizon_steps=60, seed=args.seed)

    model = PPO.load(args.model, device="cpu")
    rng = np.random.default_rng(args.seed)
    max_start = len(df_te) - 60 - 1

    rows = []
    for k in range(args.n):
        start = int(rng.integers(0, max_start))
        ppo_sf = run_ppo_fixed(model, env_ppo, start)
        vwap_sf = run_vwap_fixed(env_vwap, start)
        rows.append({
            "episode": k,
            "start_local": start,
            "start_idx_abs": split + start,
            "ppo_bps": round(ppo_sf, 2),
            "vwap_bps": round(vwap_sf, 2),
            "ppo_minus_vwap": round(ppo_sf - vwap_sf, 2),
        })

    # ---- paired table ----
    print(f"PAIRED PPO vs VWAP on {args.n} identical test windows ({args.model})\n")
    print(f"{'ep':>3} {'start_idx':>9} {'PPO bps':>9} {'VWAP bps':>9} {'PPO-VWAP':>9}")
    print("-" * 46)
    for r in rows:
        flag = "  PPO better" if r["ppo_minus_vwap"] < 0 else "  VWAP better"
        print(f"{r['episode']:>3} {r['start_idx_abs']:>9} {r['ppo_bps']:>9.2f} "
              f"{r['vwap_bps']:>9.2f} {r['ppo_minus_vwap']:>+9.2f}{flag}")

    ppo = np.array([r["ppo_bps"] for r in rows])
    vwap = np.array([r["vwap_bps"] for r in rows])
    diff = ppo - vwap
    print("-" * 46)
    print(f"{'mean':>3} {'':>9} {ppo.mean():>9.2f} {vwap.mean():>9.2f} {diff.mean():>+9.2f}")
    print(f"\nPPO wins {int((diff < 0).sum())}/{args.n} windows | "
          f"VWAP wins {int((diff > 0).sum())}/{args.n}")
    print(f"mean PPO-VWAP gap = {diff.mean():+.2f} bps (std {diff.std():.2f}); "
          f"median = {np.median(diff):+.2f} bps")
    print("(negative = PPO cheaper/better. Means are noisy on 12 windows; "
          "the per-episode pairing is the honest signal.)")

    json.dump({"model": args.model, "n": args.n, "seed": args.seed,
               "rows": rows,
               "summary": {"ppo_mean": float(ppo.mean()), "vwap_mean": float(vwap.mean()),
                           "gap_mean": float(diff.mean()), "gap_median": float(np.median(diff)),
                           "ppo_wins": int((diff < 0).sum()), "vwap_wins": int((diff > 0).sum())}},
              open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()