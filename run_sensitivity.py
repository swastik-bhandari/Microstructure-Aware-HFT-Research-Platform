"""
Impact-model sensitivity analysis
==================================

The temporary-impact model (eta * (q/depth)^alpha) is an uncalibrated
modeling choice. Rather than defending a single (eta, alpha), this sweep
tests whether the headline result — PPO beats the best classical baseline —
HOLDS ACROSS the plausible range of impact assumptions.

For each (alpha, eta) combo: train a short PPO(6-feat), evaluate it and
VWAP/TWAP on the held-out slice with the SAME impact settings, report the gap.

Usage:
    python run_sensitivity.py
    python run_sensitivity.py --timesteps 60000 --eval-episodes 200
"""

import argparse
import numpy as np
import pandas as pd

from execution_env import (ExecutionEnv, TradeTape, run_twap, run_vwap)

ALPHAS = [0.5, 0.6, 1.0]          # sqrt-law, Almgren-style 3/5, linear
ETAS = [0.1, 0.5, 2.0]            # spans 20x in impact strength


def eval_policy(model, env, n):
    sf = []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            done = term or trunc
        if "shortfall_bps" in info:
            sf.append(info["shortfall_bps"])
    return np.mean(sf)


def eval_baseline(fn, env, n):
    sf = []
    for _ in range(n):
        _, info = fn(env)
        if "shortfall_bps" in info:
            sf.append(info["shortfall_bps"])
    return np.mean(sf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="states_eth_20251201.csv")
    ap.add_argument("--tape", default="trade_tape_eth_20251201.npz")
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--target-qty", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=60)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    tape = TradeTape(args.tape)
    df = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    n = len(df)
    split = int(n * 0.70)
    df_tr, df_te = df.iloc[:split], df.iloc[split:]

    print(f"{'alpha':>6} {'eta':>6} | {'PPO6':>8} {'VWAP':>8} {'TWAP':>8} "
          f"| {'PPO-VWAP gap':>12}")
    print("-" * 60)

    rows = []
    for alpha in ALPHAS:
        for eta in ETAS:
            def mk(d, seed=None):
                return ExecutionEnv(d, None, target_qty=args.target_qty,
                                    horizon_steps=args.horizon,
                                    use_signal=False, trade_tape=tape,
                                    impact_eta=eta, impact_alpha=alpha,
                                    seed=seed)
            ppo = PPO("MlpPolicy", mk(df_tr, 0), verbose=0, device="cpu",
                      seed=0, n_steps=2048, batch_size=256)
            ppo.learn(total_timesteps=args.timesteps)

            ev = mk(df_te, 1)
            sf_ppo = eval_policy(ppo, ev, args.eval_episodes)
            sf_vwap = eval_baseline(run_vwap, mk(df_te, 2), args.eval_episodes)
            sf_twap = eval_baseline(run_twap, mk(df_te, 3), args.eval_episodes)
            gap = sf_vwap - sf_ppo
            rows.append((alpha, eta, sf_ppo, sf_vwap, sf_twap, gap))
            print(f"{alpha:>6.1f} {eta:>6.1f} | {sf_ppo:>8.2f} {sf_vwap:>8.2f} "
                  f"{sf_twap:>8.2f} | {gap:>+12.2f}")

    gaps = np.array([r[5] for r in rows])
    pd.DataFrame(rows, columns=["alpha", "eta", "ppo6_bps", "vwap_bps",
                                "twap_bps", "ppo_vs_vwap_gap_bps"]
                 ).to_csv("sensitivity_results.csv", index=False)
    print("\nSaved -> sensitivity_results.csv")
    print("\n=== SENSITIVITY SUMMARY ===")
    print(f"PPO-vs-VWAP gap across {len(rows)} impact settings:")
    print(f"  min {gaps.min():+.2f}  median {np.median(gaps):+.2f}  "
          f"max {gaps.max():+.2f} bps")
    if (gaps > 0).all():
        print("  -> PPO beats VWAP under ALL tested impact assumptions.")
    else:
        k = (gaps <= 0).sum()
        print(f"  -> PPO loses to VWAP in {k}/{len(rows)} settings — "
              f"the edge is impact-model dependent. Report honestly.")


if __name__ == "__main__":
    main()