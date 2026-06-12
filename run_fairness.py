"""
Fairness check: does an expanded (size-aware) action space let PPO beat VWAP?
============================================================================

v1 PPO had a fixed slice size; VWAP could size by volume. This script retrains
PPO with the 7-action size-aware policy (ExecutionEnvV2) under REAL trade-
attribution fills, across N seeds, and compares to the SAME baselines.

Interpretation:
  - If size-aware PPO STILL ties VWAP  -> null result is robust (the agent had
    a fair toolkit and learning still didn't beat the volume rule).
  - If size-aware PPO BEATS VWAP       -> an honest, smaller edge exists that the
    handicapped v1 action space was hiding. Report the recovered margin.

Saves fairness_results.csv + fairness_summary.json (separate filenames; does
NOT overwrite your v1 variance results).

Usage:
    python run_fairness.py --runs 5 --timesteps 100000
"""

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import joblib

from execution_env_v2 import ExecutionEnvV2
from execution_env import TradeTape, run_twap, run_vwap, run_pov, run_passive
from train_cnn_lstm_v2 import (CNNLSTM, add_directional_features,
                               FEATURES as SIGNAL_FEATURES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_signal(df, seq_len=60):
    df = add_directional_features(df)
    scaler = joblib.load("scaler_v2.joblib")
    X = scaler.transform(df[SIGNAL_FEATURES].to_numpy(dtype=np.float32))
    model = CNNLSTM(len(SIGNAL_FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load("cnn_lstm_v2.pt", map_location=DEVICE))
    model.eval()
    probs = np.full(len(df), 0.5, dtype=np.float32)
    idxs = list(range(seq_len, len(df)))
    B = 4096
    with torch.no_grad():
        for s in range(0, len(idxs), B):
            bi = idxs[s:s+B]
            seqs = np.stack([X[i-seq_len:i] for i in bi]).astype(np.float32)
            p = torch.softmax(model(torch.from_numpy(seqs).to(DEVICE)), 1)[:, 1]
            for k, i in enumerate(bi):
                probs[i] = p[k].item()
    return pd.Series(probs, index=df.index)


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
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--target-qty", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--eval-episodes", type=int, default=300)
    ap.add_argument("--tape", default="trade_tape_eth_20251201.npz")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    os.makedirs("models", exist_ok=True)

    tape = TradeTape(args.tape)
    print("Fairness check — size-aware action space, REAL fills")
    print(f"Device: {DEVICE} | runs={args.runs} | timesteps={args.timesteps}")
    df = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    signal = compute_signal(df)

    split = int(len(df) * 0.70)
    df_tr, sig_tr = df.iloc[:split], signal.iloc[:split]
    df_te, sig_te = df.iloc[split:], signal.iloc[split:]

    def mk(d, s, use_signal, seed=None):
        return ExecutionEnvV2(d, s, target_qty=args.target_qty,
                              horizon_steps=args.horizon, use_signal=use_signal,
                              trade_tape=tape, seed=seed)

    # we report the 7-feature size-aware agent (best toolkit) vs baselines,
    # and also the 6-feature size-aware agent for completeness
    sf6_runs, sf7_runs = [], []
    for run in range(args.runs):
        seed = 1000 + run
        e6 = mk(df_tr, sig_tr, False, seed)
        p6 = PPO("MlpPolicy", e6, verbose=0, device="cpu", seed=seed,
                 n_steps=2048, batch_size=256)
        p6.learn(total_timesteps=args.timesteps)
        s6 = eval_policy(p6, mk(df_te, sig_te, False, seed), args.eval_episodes)
        sf6_runs.append(s6); p6.save(f"models/ppo6_sizeaware_seed{seed}")

        e7 = mk(df_tr, sig_tr, True, seed)
        p7 = PPO("MlpPolicy", e7, verbose=0, device="cpu", seed=seed,
                 n_steps=2048, batch_size=256)
        p7.learn(total_timesteps=args.timesteps)
        s7 = eval_policy(p7, mk(df_te, sig_te, True, seed), args.eval_episodes)
        sf7_runs.append(s7); p7.save(f"models/ppo7_sizeaware_seed{seed}")

        print(f"  run {run+1}/{args.runs} (seed {seed}):  "
              f"PPO6={s6:.2f}  PPO7={s7:.2f}")

    # baselines on the SAME held-out slice (size-aware env, identical mechanics)
    bl = mk(df_te, sig_te, False, seed=42)
    twap = eval_baseline(run_twap, bl, args.eval_episodes)
    vwap = eval_baseline(run_vwap, bl, args.eval_episodes)
    pov = eval_baseline(run_pov, bl, args.eval_episodes)
    passive = eval_baseline(run_passive, bl, args.eval_episodes)

    sf6 = np.array(sf6_runs); sf7 = np.array(sf7_runs)
    best_ppo = min(sf6.mean(), sf7.mean())
    gap = vwap - best_ppo                     # positive => PPO beats VWAP

    print("\n" + "=" * 56)
    print(" FAIRNESS CHECK — shortfall (bps, lower=better)")
    print("=" * 56)
    print(f"  PPO size-aware (6 feat):  {sf6.mean():6.2f} +/- {sf6.std():.2f}")
    print(f"  PPO size-aware (7 feat):  {sf7.mean():6.2f} +/- {sf7.std():.2f}")
    print(f"  VWAP:                     {vwap:6.2f}")
    print(f"  TWAP:                     {twap:6.2f}")
    print(f"  POV:                      {pov:6.2f}")
    print(f"  Passive:                  {passive:6.2f}")
    print(f"\n  best size-aware PPO vs VWAP: {gap:+.2f} bps")
    # significance vs VWAP using the better arm's seed spread
    better = sf6 if sf6.mean() <= sf7.mean() else sf7
    se = better.std(ddof=1) / np.sqrt(len(better)) if len(better) > 1 else float("inf")
    if abs(gap) < 2 * se:
        print(f"  -> within noise (se {se:.2f}): PPO still TIES VWAP. "
              f"Null result is ROBUST to a fair action space.")
    elif gap > 0:
        print(f"  -> PPO BEATS VWAP by {gap:.2f} bps (se {se:.2f}). "
              f"Size-aware actions recover an honest edge.")
    else:
        print(f"  -> PPO worse than VWAP (se {se:.2f}).")

    seeds = [1000 + i for i in range(args.runs)]
    pd.DataFrame({"seed": seeds, "ppo6_sizeaware_bps": sf6,
                  "ppo7_sizeaware_bps": sf7}).to_csv(
        "fairness_results.csv", index=False)
    summary = {
        "experiment": "fairness_check_size_aware_action_space",
        "fill_model": "real_trade_attribution",
        "action_space": "7 actions (hold + {limit,market} x {small,medium,large})",
        "tape_file": args.tape, "states_file": args.file,
        "config": {"runs": args.runs, "timesteps": args.timesteps,
                   "target_qty": args.target_qty, "horizon": args.horizon,
                   "eval_episodes": args.eval_episodes, "seeds": seeds},
        "results": {
            "ppo6_sizeaware_mean_bps": float(sf6.mean()),
            "ppo6_sizeaware_std_bps": float(sf6.std()),
            "ppo7_sizeaware_mean_bps": float(sf7.mean()),
            "ppo7_sizeaware_std_bps": float(sf7.std()),
            "vwap_bps": vwap, "twap_bps": twap, "pov_bps": pov,
            "passive_bps": passive,
            "best_ppo_vs_vwap_bps": float(gap),
            "stderr_bps": float(se),
            "ties_vwap": bool(abs(gap) < 2 * se),
        },
    }
    with open("fairness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved -> fairness_results.csv, fairness_summary.json, "
          "models/ppo{6,7}_sizeaware_seed*.zip")


if __name__ == "__main__":
    main()
