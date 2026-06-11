"""
HFT Research Platform — Multi-run variance harness
===================================================

Trains the 6-feature and 7-feature PPO agents N times each with different
seeds, then reports mean +/- std implementation shortfall. This converts
single-run numbers (which are noisy) into statistically honest results and
tests whether the CNN-LSTM contribution is real or within noise.

Baselines (TWAP/VWAP/POV/Passive) are deterministic given the eval slices,
so they're evaluated once over many episodes.

Usage:
    python run_variance.py
    python run_variance.py --runs 5 --timesteps 100000 --eval-episodes 300
"""

import argparse
import numpy as np
import pandas as pd
import torch
import joblib

from execution_env import (ExecutionEnv, TradeTape, run_twap, run_vwap,
                           run_pov, run_passive)
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


def eval_policy(model, env, n_episodes):
    sf = []
    for _ in range(n_episodes):
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


def eval_baseline(fn, env, n_episodes):
    sf = []
    for _ in range(n_episodes):
        _, info = fn(env)
        if "shortfall_bps" in info:
            sf.append(info["shortfall_bps"])
    return np.mean(sf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="states_eth_20251201.csv")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--target-qty", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--eval-episodes", type=int, default=300)
    ap.add_argument("--tape", default="trade_tape_eth_20251201.npz",
                    help="trade tape npz for real fill attribution "
                         "(use --tape none for the old proxy)")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    import os, json
    os.makedirs("models", exist_ok=True)

    tape = None if args.tape.lower() == "none" else TradeTape(args.tape)
    print(f"Fill model: {'REAL trade attribution' if tape else 'proxy (approx)'}")
    print(f"Device: {DEVICE} | runs={args.runs} | timesteps={args.timesteps}")
    df = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    signal = compute_signal(df)

    n = len(df)
    split = int(n * 0.70)
    df_tr, sig_tr = df.iloc[:split], signal.iloc[:split]
    df_te, sig_te = df.iloc[split:], signal.iloc[split:]

    def make_env(d, s, use_signal, seed=None):
        return ExecutionEnv(d, s, target_qty=args.target_qty,
                            horizon_steps=args.horizon,
                            use_signal=use_signal, trade_tape=tape,
                            seed=seed)

    sf6_runs, sf7_runs = [], []

    for run in range(args.runs):
        seed = 1000 + run
        # --- 6 feat ---
        env6 = make_env(df_tr, sig_tr, False, seed)
        ppo6 = PPO("MlpPolicy", env6, verbose=0, device="cpu", seed=seed,
                   n_steps=2048, batch_size=256)
        ppo6.learn(total_timesteps=args.timesteps)
        sf6 = eval_policy(ppo6, make_env(df_te, sig_te, False, seed),
                          args.eval_episodes)
        sf6_runs.append(sf6)
        ppo6.save(f"models/ppo6_seed{seed}")

        # --- 7 feat ---
        env7 = make_env(df_tr, sig_tr, True, seed)
        ppo7 = PPO("MlpPolicy", env7, verbose=0, device="cpu", seed=seed,
                   n_steps=2048, batch_size=256)
        ppo7.learn(total_timesteps=args.timesteps)
        sf7 = eval_policy(ppo7, make_env(df_te, sig_te, True, seed),
                          args.eval_episodes)
        sf7_runs.append(sf7)
        ppo7.save(f"models/ppo7_seed{seed}")

        print(f"  run {run+1}/{args.runs} (seed {seed}):  "
              f"PPO6={sf6:.2f}  PPO7={sf7:.2f}  diff={sf6-sf7:+.2f}")

    # --- baselines (once, deterministic schedules over many episodes) ---
    bl = make_env(df_te, sig_te, False, seed=42)
    twap = eval_baseline(run_twap, bl, args.eval_episodes)
    vwap = eval_baseline(run_vwap, bl, args.eval_episodes)
    pov = eval_baseline(run_pov, bl, args.eval_episodes)
    passive = eval_baseline(run_passive, bl, args.eval_episodes)

    sf6_runs = np.array(sf6_runs)
    sf7_runs = np.array(sf7_runs)
    diff = sf6_runs - sf7_runs   # positive => 7-feat better

    print("\n" + "=" * 56)
    print(" RESULTS — implementation shortfall (bps, lower=better)")
    print("=" * 56)
    print(f"  PPO (6 feat):  {sf6_runs.mean():6.2f} +/- {sf6_runs.std():.2f}")
    print(f"  PPO (7 feat):  {sf7_runs.mean():6.2f} +/- {sf7_runs.std():.2f}")
    print(f"  VWAP:          {vwap:6.2f}")
    print(f"  TWAP:          {twap:6.2f}")
    print(f"  POV:           {pov:6.2f}")
    print(f"  Passive:       {passive:6.2f}")

    print("\n--- CNN-LSTM contribution (PPO6 - PPO7) ---")
    print(f"  mean diff: {diff.mean():+.2f} bps  (std {diff.std():.2f})")
    # simple significance: is mean diff bigger than its own std error?
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("inf")
    print(f"  std error: {se:.2f} bps")
    if abs(diff.mean()) < 2 * se:
        print("  -> NOT significant: signal contribution is within noise.")
    else:
        sign = "helps" if diff.mean() > 0 else "HURTS"
        print(f"  -> significant: CNN-LSTM signal {sign} execution.")

    print("\n--- PPO vs best classical baseline ---")
    best_classical = min(twap, vwap, pov)
    best_ppo = min(sf6_runs.mean(), sf7_runs.mean())
    print(f"  best PPO {best_ppo:.2f} vs best classical (VWAP/TWAP/POV) "
          f"{best_classical:.2f}  ->  {best_classical - best_ppo:+.2f} bps")

    # ---- persist everything (research evidence) ----
    seeds = [1000 + i for i in range(args.runs)]
    pd.DataFrame({"seed": seeds,
                  "ppo6_shortfall_bps": sf6_runs,
                  "ppo7_shortfall_bps": sf7_runs,
                  "diff_6_minus_7": diff}).to_csv(
        "variance_results.csv", index=False)

    summary = {
        "fill_model": "real_trade_attribution" if tape else "proxy",
        "tape_file": args.tape,
        "states_file": args.file,
        "config": {"runs": args.runs, "timesteps": args.timesteps,
                   "target_qty": args.target_qty, "horizon": args.horizon,
                   "eval_episodes": args.eval_episodes, "seeds": seeds,
                   "ppo": {"policy": "MlpPolicy", "n_steps": 2048,
                           "batch_size": 256, "device": "cpu"}},
        "results": {
            "ppo6_mean_bps": float(sf6_runs.mean()),
            "ppo6_std_bps": float(sf6_runs.std()),
            "ppo7_mean_bps": float(sf7_runs.mean()),
            "ppo7_std_bps": float(sf7_runs.std()),
            "vwap_bps": float(vwap), "twap_bps": float(twap),
            "pov_bps": float(pov), "passive_bps": float(passive),
            "cnn_lstm_contribution_mean_bps": float(diff.mean()),
            "cnn_lstm_contribution_stderr_bps": float(se),
            "cnn_lstm_significant": bool(abs(diff.mean()) >= 2 * se),
            "ppo_vs_best_classical_bps": float(best_classical - best_ppo),
        },
    }
    with open("variance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved -> variance_results.csv, variance_summary.json, "
          "models/ppo{6,7}_seed*.zip")


if __name__ == "__main__":
    main()