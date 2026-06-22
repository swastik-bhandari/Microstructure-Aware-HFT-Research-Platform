"""Run a trained PPO agent through real episodes and record per-timestep actions
for the web sim. Honest: these are the actual agent's decisions, replayed."""
import json, numpy as np, pandas as pd, joblib, torch
from stable_baselines3 import PPO
from execution_env import ExecutionEnv, TradeTape, MARKET_FEATURES
from train_cnn_lstm_v2 import CNNLSTM, add_directional_features
from run_variance import compute_signal

DEVICE="cpu"
ACTNAME={0:"HOLD",1:"LIMIT",2:"MARKET"}

df = pd.read_csv("states_eth_20251201.csv").sort_values("ts").reset_index(drop=True)
tape = TradeTape("trade_tape_eth_20251201.npz")
signal = compute_signal(df)                      # P(up), frozen CNN-LSTM

# use the 7-feature agent so we also show P(up) feeding it
model = PPO.load("models/ppo7_seed1000.zip", device=DEVICE)

def run_episode(start):
    env = ExecutionEnv(df, signal, use_signal=True, trade_tape=tape,
                       target_qty=10.0, horizon_steps=60, seed=0)
    env.reset()
    # pin a chosen start window (mirrors reset()'s own assignments)
    env.start = start; env.t = start
    env.steps_left = env.H; env.inventory = env.target_qty
    env.arrival_mid = env.mid[start]; env.realized_value = 0.0; env.executed_qty = 0.0
    obs = env._obs()
    rows=[]
    for step in range(env.H):
        a,_ = model.predict(obs, deterministic=True)
        a=int(a)
        inv_before=env.inventory
        t_now=env.t
        obs, r, term, trunc, info = env.step(a)
        sold = inv_before - env.inventory
        avg = (env.realized_value/env.executed_qty) if env.executed_qty>1e-9 else env.arrival_mid
        sf = (env.arrival_mid - avg)/env.arrival_mid*1e4 if env.executed_qty>1e-9 else 0.0
        rows.append({
            "step":step, "t":int(t_now),
            "mid":round(float(env.mid[t_now]),2),
            "action":ACTNAME[a],
            "sold":round(float(sold),3),
            "inv":round(float(env.inventory),3),
            "pup":round(float(signal.iloc[t_now]),3),
            "shortfall":round(float(sf),2),
        })
        if term or trunc: break
    return {"start":int(start),"arrival":round(env.arrival_mid,2),
            "final_sf":round(float(sf),2),"rows":rows}

# pick 3 windows: calm, the selloff (~t50000), recovery (~t78000)
episodes=[run_episode(s) for s in [20000, 51000, 78000]]
json.dump(episodes, open("ppo_episodes.json","w"), separators=(",",":"))
print("saved ppo_episodes.json")
for e in episodes:
    acts=[r["action"] for r in e["rows"]]
    from collections import Counter
    print(f"  start t={e['start']}: {dict(Counter(acts))} | final shortfall {e['final_sf']} bps")
