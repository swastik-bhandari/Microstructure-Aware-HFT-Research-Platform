"""
Outlier re-test: isolate one (alpha, eta) impact cell and run it properly.
=========================================================================

The sensitivity sweep flagged (alpha=0.5, eta=0.5) at -25.7 bps vs VWAP, far
from its smooth neighbours. Hypothesis: a single short-training PPO run failed
to converge, not a real impact effect. This script retrains that ONE cell with
more timesteps and multiple seeds to see if it regresses to the ~tie its
neighbours show.

Usage (defaults target the outlier):
    python retest_cell.py --alpha 0.5 --eta 0.5 --seeds 5 --timesteps 120000
"""

import argparse, json
import numpy as np
import pandas as pd

from execution_env import ExecutionEnv, TradeTape, run_twap, run_vwap


def eval_policy(model, env, n):
    sf = []
    for _ in range(n):
        obs, _ = env.reset(); done = False; info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a)); done = term or trunc
        if "shortfall_bps" in info: sf.append(info["shortfall_bps"])
    return float(np.mean(sf))


def eval_baseline(fn, env, n):
    sf = []
    for _ in range(n):
        _, info = fn(env)
        if "shortfall_bps" in info: sf.append(info["shortfall_bps"])
    return float(np.mean(sf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="states_eth_20251201.csv")
    ap.add_argument("--tape", default="trade_tape_eth_20251201.npz")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=120_000)
    ap.add_argument("--eval-episodes", type=int, default=300)
    ap.add_argument("--target-qty", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=60)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    tape = TradeTape(args.tape)
    df = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    split = int(len(df) * 0.70)
    df_tr, df_te = df.iloc[:split], df.iloc[split:]

    def mk(d, seed=None):
        return ExecutionEnv(d, None, target_qty=args.target_qty,
                            horizon_steps=args.horizon, use_signal=False,
                            trade_tape=tape, impact_eta=args.eta,
                            impact_alpha=args.alpha, seed=seed)

    print(f"Re-testing cell alpha={args.alpha} eta={args.eta} "
          f"with {args.seeds} seeds x {args.timesteps} steps")
    print("-" * 50)

    ppo_runs = []
    for s in range(args.seeds):
        seed = 2000 + s
        ppo = PPO("MlpPolicy", mk(df_tr, seed), verbose=0, device="cpu",
                  seed=seed, n_steps=2048, batch_size=256)
        ppo.learn(total_timesteps=args.timesteps)
        sf = eval_policy(ppo, mk(df_te, seed), args.eval_episodes)
        ppo_runs.append(sf)
        print(f"  seed {seed}:  PPO={sf:.2f}")

    vwap = eval_baseline(run_vwap, mk(df_te, 42), args.eval_episodes)
    twap = eval_baseline(run_twap, mk(df_te, 43), args.eval_episodes)

    ppo_runs = np.array(ppo_runs)
    gap = vwap - ppo_runs.mean()
    se = ppo_runs.std(ddof=1) / np.sqrt(len(ppo_runs)) if len(ppo_runs) > 1 else float("inf")

    print("\n" + "=" * 50)
    print(f"  PPO:   {ppo_runs.mean():.2f} +/- {ppo_runs.std():.2f}  "
          f"(min {ppo_runs.min():.2f}, max {ppo_runs.max():.2f})")
    print(f"  VWAP:  {vwap:.2f}")
    print(f"  TWAP:  {twap:.2f}")
    print(f"  gap (VWAP - PPO): {gap:+.2f} bps")
    print("=" * 50)
    if abs(gap) < 2 * se:
        print("  -> Converges to a TIE. The original -25.7 was a short-training")
        print("     convergence failure. Footnote it as an artifact.")
    else:
        print(f"  -> Gap persists ({gap:+.2f}). Not just noise — investigate")
        print("     why this impact regime behaves differently.")

    verdict = ("tie_converged_artifact" if abs(gap) < 2 * se
               else "gap_persists_real_effect")
    with open(f"retest_alpha{args.alpha}_eta{args.eta}_summary.json", "w") as f:
        json.dump({
            "cell": {"alpha": args.alpha, "eta": args.eta},
            "config": {"seeds": args.seeds, "timesteps": args.timesteps,
                       "eval_episodes": args.eval_episodes},
            "ppo_mean_bps": float(ppo_runs.mean()),
            "ppo_std_bps": float(ppo_runs.std()),
            "vwap_bps": vwap, "twap_bps": twap,
            "gap_vwap_minus_ppo_bps": float(gap),
            "stderr_bps": float(se),
            "verdict": verdict,
            "original_sweep_gap_bps": -25.71,
        }, f, indent=2)

    pd.DataFrame({"seed": [2000 + i for i in range(args.seeds)],
                  "ppo_bps": ppo_runs}).to_csv(
        f"retest_alpha{args.alpha}_eta{args.eta}.csv", index=False)
    print(f"\nSaved -> retest_alpha{args.alpha}_eta{args.eta}.csv")


if __name__ == "__main__":
    main()