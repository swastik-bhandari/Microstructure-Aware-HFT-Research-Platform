"""
run_variance_triple.py
======================
Full statistical comparison: PPO_static vs PPO_sizeaware vs VWAP
300 episodes × 5 seeds = 1,500 paired windows per agent.

Design mirrors run_variance.py exactly:
  - same 70/30 chronological split (split = int(n * 0.70))
  - same seeds 1000-1004
  - same eval_episodes = 300
  - same TradeTape real fills
  - PAIRED: all three agents run on the SAME start points per seed

For each seed k:
  1. load ppo7_seed{k}.zip      -> ExecutionEnv  (3-action, fixed-size)
  2. load ppo7_sizeaware_seed{k}.zip -> ExecutionEnvV2 (7-action, size-aware)
  3. VWAP                        -> ExecutionEnv  (qty_override, env-agnostic)
  4. sample 300 random test starts with rng(seed=k)
  5. run all three on each start, record shortfall_bps

Outputs:
  variance_triple_raw.csv    -- all 1500 rows (episode, seed, all 3 bps)
  variance_triple_seeds.csv  -- per-seed summary (mean/median/std per agent)
  variance_triple_summary.json -- full stats: per-seed + aggregate (mean of means + pooled)

Usage:
    python run_variance_triple.py                     # default: n=300, seeds 1000-1004
    python run_variance_triple.py --n 10 --seeds 1000 1001  # quick smoke-test
"""
import argparse, json, csv, time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from execution_env import ExecutionEnv, TradeTape
from execution_env_v2 import ExecutionEnvV2
from run_variance import compute_signal


# ── env helpers ──────────────────────────────────────────────────────────────

def reset_to(env, start):
    """Pin the env to a fixed start point, bypassing the env's internal rng."""
    env.reset()
    env.start          = start
    env.t              = start
    env.steps_left     = env.H
    env.inventory      = env.target_qty
    env.arrival_mid    = env.mid[start]
    env.realized_value = 0.0
    env.executed_qty   = 0.0
    return env._obs()


def run_ppo_ep(model, env, start):
    """One deterministic PPO episode from a fixed start. Returns shortfall_bps."""
    obs = reset_to(env, start)
    done, info = False, {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
    sf = info.get("shortfall_bps", np.nan)
    return float(sf)


def run_vwap_ep(env, start):
    """One VWAP episode from a fixed start. Returns shortfall_bps."""
    reset_to(env, start)
    H   = env.H
    vol = env.intensity[start:start + H].astype(np.float64)
    vol = np.where(vol > 0, vol, max(float(vol.mean()), 1e-9))
    weights = vol / vol.sum()
    done, info, step_i = False, {}, 0
    while not done:
        qty = env.target_qty * weights[min(step_i, H - 1)]
        _, _, term, trunc, info = env.step(2, qty_override=qty)
        done = term or trunc
        step_i += 1
    sf = info.get("shortfall_bps", np.nan)
    return float(sf)


# ── stats helpers ─────────────────────────────────────────────────────────────

def summarise(arr):
    a = np.array(arr, dtype=float)
    return {
        "mean":   round(float(a.mean()),   3),
        "median": round(float(np.median(a)), 3),
        "std":    round(float(a.std()),    3),
        "min":    round(float(a.min()),    3),
        "max":    round(float(a.max()),    3),
        "n":      len(a),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file",    default="states_eth_20251201.csv")
    ap.add_argument("--tape",    default="trade_tape_eth_20251201.npz")
    ap.add_argument("--n",       type=int, default=300, help="episodes per seed")
    ap.add_argument("--seeds",   type=int, nargs="+", default=[1000,1001,1002,1003,1004])
    ap.add_argument("--split",   type=float, default=0.70)
    ap.add_argument("--out-raw",   default="variance_triple_raw.csv")
    ap.add_argument("--out-seeds", default="variance_triple_seeds.csv")
    ap.add_argument("--out-json",  default="variance_triple_summary.json")
    args = ap.parse_args()

    # ── data (loaded once) ───────────────────────────────────────────────────
    print("loading data...")
    df     = pd.read_csv(args.file).sort_values("ts").reset_index(drop=True)
    signal = compute_signal(df)
    tape   = TradeTape(args.tape)
    n_tot  = len(df)
    split  = int(n_tot * args.split)
    df_te  = df.iloc[split:].reset_index(drop=True)
    sig_te = signal.iloc[split:].reset_index(drop=True)
    max_start = len(df_te) - 60 - 1
    print(f"states={n_tot} | split={split} | test={n_tot-split} ({(n_tot-split)/3600:.1f}h)")
    print(f"plan: {len(args.seeds)} seeds × {args.n} episodes = "
          f"{len(args.seeds)*args.n} windows per agent\n")

    # ── output files ─────────────────────────────────────────────────────────
    raw_fields = ["seed","episode","start_local","start_idx_abs",
                  "static_bps","sizeaware_bps","vwap_bps",
                  "static_vs_vwap","sizeaware_vs_vwap","sizeaware_vs_static"]
    raw_f   = open(args.out_raw,   "w", newline="")
    seeds_f = open(args.out_seeds, "w", newline="")
    raw_w   = csv.DictWriter(raw_f,   fieldnames=raw_fields);   raw_w.writeheader()
    seed_w  = csv.DictWriter(seeds_f, fieldnames=[
        "seed","static_mean","static_median","static_std",
        "sizeaware_mean","sizeaware_median","sizeaware_std",
        "vwap_mean","vwap_median","vwap_std",
        "static_beats_vwap","sizeaware_beats_vwap","sizeaware_beats_static"]);
    seed_w.writeheader()

    # ── collect per-seed means for the "mean of means" aggregate ─────────────
    seed_means = {"static": [], "sizeaware": [], "vwap": []}
    all_static, all_sa, all_vwap = [], [], []
    all_sv, all_sav, all_sas = [], [], []

    t0 = time.time()

    for seed in args.seeds:
        print(f"── seed {seed} ──────────────────────────────────────────────")

        # load the two PPO models for this seed
        static_path = f"models/ppo7_seed{seed}.zip"
        sa_path     = f"models/ppo7_sizeaware_seed{seed}.zip"
        m_static = PPO.load(static_path, device="cpu")
        m_sa     = PPO.load(sa_path,     device="cpu")
        print(f"  loaded {static_path}  +  {sa_path}")

        # build envs on the test slice
        env_st = ExecutionEnv( df_te, sig_te, use_signal=True,  trade_tape=tape,
                               target_qty=10.0, horizon_steps=60, seed=seed)
        env_sa = ExecutionEnvV2(df_te, sig_te, use_signal=True, trade_tape=tape,
                                target_qty=10.0, horizon_steps=60, seed=seed)
        env_vw = ExecutionEnv( df_te, sig_te, use_signal=False, trade_tape=tape,
                               target_qty=10.0, horizon_steps=60, seed=seed)

        # sample n random start points for this seed
        rng    = np.random.default_rng(seed)
        starts = [int(rng.integers(0, max_start)) for _ in range(args.n)]

        st_bps, sa_bps, vw_bps = [], [], []

        for ep, start in enumerate(starts):
            sf_st = run_ppo_ep(m_static, env_st, start)
            sf_sa = run_ppo_ep(m_sa,     env_sa, start)
            sf_vw = run_vwap_ep(env_vw,          start)

            st_bps.append(sf_st); sa_bps.append(sf_sa); vw_bps.append(sf_vw)

            raw_w.writerow({
                "seed": seed, "episode": ep,
                "start_local": start, "start_idx_abs": split + start,
                "static_bps":       round(sf_st, 3),
                "sizeaware_bps":    round(sf_sa, 3),
                "vwap_bps":         round(sf_vw, 3),
                "static_vs_vwap":   round(sf_st - sf_vw, 3),
                "sizeaware_vs_vwap": round(sf_sa - sf_vw, 3),
                "sizeaware_vs_static": round(sf_sa - sf_st, 3),
            })

            if (ep + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  ep {ep+1:>3}/{args.n} | "
                      f"static {np.mean(st_bps):.1f} | "
                      f"sizeaware {np.mean(sa_bps):.1f} | "
                      f"vwap {np.mean(vw_bps):.1f} bps  [{elapsed:.0f}s]")

        raw_f.flush()

        # per-seed stats
        sv  = np.array(st_bps) - np.array(vw_bps)
        sav = np.array(sa_bps) - np.array(vw_bps)
        sas = np.array(sa_bps) - np.array(st_bps)

        static_beats_vwap    = int((sv  < 0).sum())
        sizeaware_beats_vwap = int((sav < 0).sum())
        sizeaware_beats_static = int((sas < 0).sum())

        seed_w.writerow({
            "seed": seed,
            "static_mean":   round(float(np.mean(st_bps)), 3),
            "static_median": round(float(np.median(st_bps)), 3),
            "static_std":    round(float(np.std(st_bps)), 3),
            "sizeaware_mean":   round(float(np.mean(sa_bps)), 3),
            "sizeaware_median": round(float(np.median(sa_bps)), 3),
            "sizeaware_std":    round(float(np.std(sa_bps)), 3),
            "vwap_mean":   round(float(np.mean(vw_bps)), 3),
            "vwap_median": round(float(np.median(vw_bps)), 3),
            "vwap_std":    round(float(np.std(vw_bps)), 3),
            "static_beats_vwap":    static_beats_vwap,
            "sizeaware_beats_vwap": sizeaware_beats_vwap,
            "sizeaware_beats_static": sizeaware_beats_static,
        })
        seeds_f.flush()

        # accumulate
        seed_means["static"].append(float(np.mean(st_bps)))
        seed_means["sizeaware"].append(float(np.mean(sa_bps)))
        seed_means["vwap"].append(float(np.mean(vw_bps)))
        all_static.extend(st_bps); all_sa.extend(sa_bps); all_vwap.extend(vw_bps)
        all_sv.extend(sv.tolist()); all_sav.extend(sav.tolist()); all_sas.extend(sas.tolist())

        print(f"  seed {seed} DONE | "
              f"static {np.mean(st_bps):.2f} | "
              f"sizeaware {np.mean(sa_bps):.2f} | "
              f"vwap {np.mean(vw_bps):.2f} | "
              f"sa beats vwap {sizeaware_beats_vwap}/{args.n}")

    raw_f.close(); seeds_f.close()
    total_time = time.time() - t0

    # ── aggregate summary ─────────────────────────────────────────────────────
    N_total = len(args.seeds) * args.n

    # mean of per-seed means (matches run_variance.py's aggregation method)
    mom_static   = float(np.mean(seed_means["static"]))
    mom_sa       = float(np.mean(seed_means["sizeaware"]))
    mom_vwap     = float(np.mean(seed_means["vwap"]))

    summary = {
        "study": {
            "seeds": args.seeds,
            "episodes_per_seed": args.n,
            "total_windows": N_total,
            "split": split,
            "test_states": n_tot - split,
            "runtime_seconds": round(total_time, 1),
        },
        "mean_of_seed_means": {
            "static_bps":    round(mom_static, 3),
            "sizeaware_bps": round(mom_sa,     3),
            "vwap_bps":      round(mom_vwap,   3),
        },
        "pooled_1500": {
            "static":    summarise(all_static),
            "sizeaware": summarise(all_sa),
            "vwap":      summarise(all_vwap),
        },
        "gaps_pooled": {
            "static_vs_vwap":      summarise(all_sv),
            "sizeaware_vs_vwap":   summarise(all_sav),
            "sizeaware_vs_static": summarise(all_sas),
        },
        "win_rates_pooled": {
            "static_beats_vwap":       {
                "count": int(np.sum(np.array(all_sv) < 0)),
                "pct":   round(100 * np.mean(np.array(all_sv) < 0), 1)},
            "sizeaware_beats_vwap":    {
                "count": int(np.sum(np.array(all_sav) < 0)),
                "pct":   round(100 * np.mean(np.array(all_sav) < 0), 1)},
            "sizeaware_beats_static":  {
                "count": int(np.sum(np.array(all_sas) < 0)),
                "pct":   round(100 * np.mean(np.array(all_sas) < 0), 1)},
        },
        "per_seed_means": seed_means,
    }

    json.dump(summary, open(args.out_json, "w"), indent=2)

    # ── final print ───────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"TRIPLE COMPARISON  |  {len(args.seeds)} seeds × {args.n} ep = {N_total} windows")
    print(f"{'='*64}")
    print(f"{'':30} {'STATIC':>10} {'SIZEAWARE':>10} {'VWAP':>10}")
    print(f"{'mean of seed means':30} {mom_static:>10.2f} {mom_sa:>10.2f} {mom_vwap:>10.2f}")
    print(f"{'pooled mean':30} {np.mean(all_static):>10.2f} {np.mean(all_sa):>10.2f} {np.mean(all_vwap):>10.2f}")
    print(f"{'pooled median':30} {np.median(all_static):>10.2f} {np.median(all_sa):>10.2f} {np.median(all_vwap):>10.2f}")
    print(f"{'pooled std':30} {np.std(all_static):>10.2f} {np.std(all_sa):>10.2f} {np.std(all_vwap):>10.2f}")
    print(f"{'pooled min':30} {np.min(all_static):>10.2f} {np.min(all_sa):>10.2f} {np.min(all_vwap):>10.2f}")
    print(f"{'pooled max':30} {np.max(all_static):>10.2f} {np.max(all_sa):>10.2f} {np.max(all_vwap):>10.2f}")
    print(f"{'beats VWAP (win rate)':30} "
          f"{summary['win_rates_pooled']['static_beats_vwap']['pct']:>9.1f}% "
          f"{summary['win_rates_pooled']['sizeaware_beats_vwap']['pct']:>9.1f}%")
    print(f"{'sizeaware beats static':30} "
          f"{'':>10} "
          f"{summary['win_rates_pooled']['sizeaware_beats_static']['pct']:>9.1f}%")
    print(f"\nruntime: {total_time:.0f}s")
    print(f"\nsaved -> {args.out_raw}")
    print(f"saved -> {args.out_seeds}")
    print(f"saved -> {args.out_json}")


if __name__ == "__main__":
    main()
