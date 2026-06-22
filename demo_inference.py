"""
Inference wrapper for the demo backend.
=======================================

Bundles the trained PPO agent + CNN-LSTM signal + scaler + execution env
into a single object the Flask app can call. Handles:
  - loading all saved artifacts
  - precomputing the P(up) signal over the state table
  - running a full execution episode (PPO) and returning step-by-step trace
  - running VWAP/TWAP on the SAME slice for side-by-side comparison
  - toggling between REAL and IDEALIZED fill models

Nothing here retrains — it only loads and runs what you already saved.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib

from execution_env import (ExecutionEnv, TradeTape, run_vwap, run_twap,
                           MARKET_FEATURES)
from train_cnn_lstm_v2 import (CNNLSTM, add_directional_features,
                               FEATURES as SIGNAL_FEATURES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ExecutionDemo:
    def __init__(self,
                 states_csv="states_eth_20251201.csv",
                 tape_npz="trade_tape_eth_20251201.npz",
                 cnn_path="cnn_lstm_v2.pt",
                 scaler_path="scaler_v2.joblib",
                 ppo6_path="models/ppo6_seed1000",
                 ppo7_path="models/ppo7_seed1000",
                 seq_len=60):
        from stable_baselines3 import PPO

        self.seq_len = seq_len
        self.df = pd.read_csv(states_csv).sort_values("ts").reset_index(drop=True)
        self.tape = TradeTape(tape_npz)

        # precompute P(up) signal across the whole table
        self.signal = self._compute_signal(cnn_path, scaler_path)

        # load both agents (6-feat and 7-feat)
        self.ppo6 = PPO.load(ppo6_path, device="cpu")
        self.ppo7 = PPO.load(ppo7_path, device="cpu")

        # cache arrays for fast book/feature serving
        self.mid = self.df["mid"].to_numpy()
        self.ts_ns = self.df["ts"].astype("int64").to_numpy()

    # -- signal -----------------------------------------------------------

    def _compute_signal(self, cnn_path, scaler_path):
        d = add_directional_features(self.df)
        scaler = joblib.load(scaler_path)
        X = scaler.transform(d[SIGNAL_FEATURES].to_numpy(dtype=np.float32))
        model = CNNLSTM(len(SIGNAL_FEATURES)).to(DEVICE)
        model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        model.eval()
        probs = np.full(len(d), 0.5, dtype=np.float32)
        idxs = list(range(self.seq_len, len(d)))
        B = 4096
        with torch.no_grad():
            for s in range(0, len(idxs), B):
                bi = idxs[s:s+B]
                seqs = np.stack([X[i-self.seq_len:i] for i in bi]).astype(np.float32)
                p = torch.softmax(model(torch.from_numpy(seqs).to(DEVICE)), 1)[:, 1]
                for k, i in enumerate(bi):
                    probs[i] = p[k].item()
        return pd.Series(probs, index=d.index)

    # -- episode ----------------------------------------------------------

    def run_episode(self, start=None, use_signal=True, real_fills=True,
                    target_qty=10.0, horizon=60, seed=0):
        """Run one PPO execution episode; return a step-by-step trace plus
        VWAP/TWAP shortfall on the same slice."""
        tape = self.tape if real_fills else None
        agent = self.ppo7 if use_signal else self.ppo6

        # fixed start so the comparison is apples-to-apples
        env = ExecutionEnv(self.df, self.signal, target_qty=target_qty,
                           horizon_steps=horizon, use_signal=use_signal,
                           trade_tape=tape, seed=seed)
        if start is not None:
            # force a specific start index for a deterministic demo slice
            env.rng = np.random.default_rng(seed)
            obs, _ = env.reset()
            env.start = int(start)
            env.t = int(start)
            env.steps_left = horizon
            env.inventory = target_qty
            env.arrival_mid = self.mid[int(start)]
            env.realized_value = 0.0
            env.executed_qty = 0.0
            obs = env._obs()
        else:
            obs, _ = env.reset()
            start = env.start

        action_names = {0: "HOLD", 1: "LIMIT", 2: "MARKET"}
        trace = []
        done = False
        step = 0
        while not done:
            t_idx = env.t
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
            inv_before = env.inventory
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            trace.append({
                "step": step,
                "t_index": int(t_idx),
                "mid": float(self.mid[min(t_idx, len(self.mid)-1)]),
                "action": action_names[action],
                "inventory_before": float(inv_before),
                "inventory_after": float(env.inventory),
                "reward": float(r),
                "p_up": float(self.signal.iloc[min(t_idx, len(self.signal)-1)]),
            })
            step += 1

        ppo_shortfall = info.get("shortfall_bps", None)

        # baselines on the same slice
        def fresh():
            e = ExecutionEnv(self.df, self.signal, target_qty=target_qty,
                             horizon_steps=horizon, use_signal=False,
                             trade_tape=tape, seed=seed)
            e.rng = np.random.default_rng(seed)
            e.reset()
            e.start = int(start); e.t = int(start)
            e.steps_left = horizon; e.inventory = target_qty
            e.arrival_mid = self.mid[int(start)]
            e.realized_value = 0.0; e.executed_qty = 0.0
            return e

        _, vwap_info = run_vwap(fresh())
        _, twap_info = run_twap(fresh())

        return {
            "start_index": int(start),
            "use_signal": use_signal,
            "real_fills": real_fills,
            "target_qty": target_qty,
            "horizon": horizon,
            "trace": trace,
            "ppo_shortfall_bps": ppo_shortfall,
            "vwap_shortfall_bps": vwap_info.get("shortfall_bps"),
            "twap_shortfall_bps": twap_info.get("shortfall_bps"),
        }

    # -- book / features at a time index ----------------------------------

    def state_at(self, t_index):
        """Return features + a synthetic depth ladder for visualization.

        We don't store the full book per step (too big), so we render a
        plausible ladder around mid from spread/depth/obi for the animation.
        The mid, spread, and features are REAL; the per-level split is a
        visualization aid derived from OBI + depth.
        """
        t = int(np.clip(t_index, 0, len(self.df) - 1))
        row = self.df.iloc[t]
        mid = float(row["mid"]); spread = float(row["spread"])
        depth = float(row["depth"]); obi = float(row["obi"])
        bid_px = mid - spread / 2
        ask_px = mid + spread / 2
        # split total depth by OBI, spread across 5 levels with decay
        bid_total = depth * obi
        ask_total = depth * (1 - obi)
        decay = np.array([0.35, 0.25, 0.18, 0.13, 0.09])
        tick = max(spread, 0.1)
        bids = [{"px": round(bid_px - i * tick, 2),
                 "sz": round(float(bid_total * decay[i]), 3)} for i in range(5)]
        asks = [{"px": round(ask_px + i * tick, 2),
                 "sz": round(float(ask_total * decay[i]), 3)} for i in range(5)]
        return {
            "t_index": t,
            "ts_ns": int(self.ts_ns[t]),
            "mid": mid, "spread": round(spread, 4),
            "features": {
                "spread": round(spread, 4),
                "obi": round(obi, 4),
                "depth": round(depth, 2),
                "volatility": round(float(row["volatility"]), 6),
                "trade_intensity": round(float(row["trade_intensity"]), 2),
                "queue_at_best": round(float(row["queue_at_best"]), 2),
            },
            "p_up": round(float(self.signal.iloc[t]), 4),
            "bids": bids, "asks": asks,
        }

    # -- static results / metadata ----------------------------------------

    def summary(self):
        out = {}
        for fn in ("variance_summary.json", "cnn_lstm_v2_meta.json"):
            if os.path.exists(fn):
                with open(fn) as f:
                    out[fn.replace(".json", "")] = json.load(f)
        if os.path.exists("sensitivity_results.csv"):
            out["sensitivity"] = pd.read_csv(
                "sensitivity_results.csv").to_dict(orient="records")
        out["data_provenance"] = {
            "n_states": int(len(self.df)),
            "n_trades": int(len(self.tape.ts)),
            "mid_min": float(self.mid.min()),
            "mid_max": float(self.mid.max()),
            "coin": "ETH", "date": "2025-12-01", "source": "Hyperliquid L4",
        }
        return out


if __name__ == "__main__":
    # quick self-test
    print("Loading demo (this loads PPO + CNN-LSTM + scaler)...")
    demo = ExecutionDemo()
    print("OK. Running one episode (real fills, 7-feat)...")
    ep = demo.run_episode(start=80000, use_signal=True, real_fills=True)
    print(f"  steps: {len(ep['trace'])}")
    print(f"  PPO shortfall:  {ep['ppo_shortfall_bps']:.2f} bps")
    print(f"  VWAP shortfall: {ep['vwap_shortfall_bps']:.2f} bps")
    print(f"  TWAP shortfall: {ep['twap_shortfall_bps']:.2f} bps")
    print(f"  first 3 actions: {[s['action'] for s in ep['trace'][:3]]}")
    print("\nState at t=80000:")
    s = demo.state_at(80000)
    print(f"  mid={s['mid']}  spread={s['spread']}  p_up={s['p_up']}")
    print(f"  best bid {s['bids'][0]}  best ask {s['asks'][0]}")
    print("\nSelf-test passed.")
