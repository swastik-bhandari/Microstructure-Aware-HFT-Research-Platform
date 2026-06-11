"""
HFT Research Platform — Layer 3b: Execution Environment (Gymnasium)
====================================================================

Optimal execution task: SELL a fixed quantity of ETH within a fixed time
window, minimizing implementation shortfall.

At each step the agent observes the market state + its own (inventory,
time) and chooses:
    0 = HOLD            (wait, place nothing)
    1 = LIMIT sell      (post passively at best ask; may or may not fill)
    2 = MARKET sell     (cross the spread; certain fill, pay spread+impact)

Reward = negative implementation shortfall for the slice executed this step,
plus a terminal penalty for any inventory force-liquidated at the end.

The environment is driven by the precomputed state table (states_*.csv) so
training is a fast replay over real microstructure — no live matching needed.

Ablation: set use_signal=False to drop the CNN-LSTM P(up) feature, isolating
its contribution to execution quality.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

MARKET_FEATURES = ["spread", "obi", "depth", "volatility",
                   "trade_intensity", "queue_at_best"]


class TradeTape:
    """Time-sorted real trades for fill attribution.

    A resting SELL limit at price P during [t0, t1) is fillable by the
    buy-aggressor ("B") trades in that window with px >= P: a buyer who
    lifted the book at or through P would have hit our order. Fill size
    is capped by the real aggressor volume that traded at/above P.
    (Level 1 attribution: price-based; queue position not modeled —
    documented as future work / Level 2.)
    """

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.ts = d["ts_ns"]          # sorted int64
        self.px = d["px"]
        self.sz = d["sz"]
        self.isB = d["isB"]

    def sell_fillable(self, t0_ns, t1_ns, limit_px):
        """Total buy-aggressor volume at px >= limit_px in [t0, t1)."""
        i0 = np.searchsorted(self.ts, t0_ns, side="left")
        i1 = np.searchsorted(self.ts, t1_ns, side="left")
        if i1 <= i0:
            return 0.0
        m = self.isB[i0:i1] & (self.px[i0:i1] >= limit_px)
        return float(self.sz[i0:i1][m].sum())


class ExecutionEnv(gym.Env):
    """Sell `target_qty` ETH over `horizon_steps` decision steps."""

    metadata = {"render_modes": []}

    def __init__(self, states_df, signal_series=None,
                 target_qty=10.0, horizon_steps=60,
                 use_signal=True, trade_tape=None,
                 impact_eta=0.5, impact_alpha=0.5, fee_bps=2.0,
                 seed=None):
        super().__init__()
        self.df = states_df.reset_index(drop=True)
        self.signal = (signal_series.reset_index(drop=True)
                       if signal_series is not None else None)
        self.use_signal = use_signal and (self.signal is not None)

        self.target_qty = target_qty
        self.H = horizon_steps
        # Temporary impact (market orders): impact_frac = eta * (q/depth)^alpha
        # UNCALIBRATED — exponent/coefficient are swept in sensitivity analysis.
        self.impact_eta = impact_eta
        self.impact_alpha = impact_alpha
        self.fee_bps = fee_bps                # taker fee, basis points
        self.tape = trade_tape                # real-fill attribution (None = proxy)
        self.rng = np.random.default_rng(seed)

        # state timestamps (ns) for aligning trade buckets to steps
        self.ts_ns = self.df["ts"].astype("int64").to_numpy()

        # scale features for the policy net (z-score over the whole table)
        feats = self.df[MARKET_FEATURES].to_numpy(dtype=np.float32)
        self.f_mean = feats.mean(0)
        self.f_std = feats.std(0) + 1e-8
        self._feats_scaled = (feats - self.f_mean) / self.f_std

        self.mid = self.df["mid"].to_numpy(dtype=np.float64)
        self.spread = self.df["spread"].to_numpy(dtype=np.float64)
        self.depth = self.df["depth"].to_numpy(dtype=np.float64)
        self.intensity = self.df["trade_intensity"].to_numpy(dtype=np.float64)

        n_market = len(MARKET_FEATURES) + (1 if self.use_signal else 0)
        # obs = market features (+signal) + [inv_frac, time_frac]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_market + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)   # hold / limit / market

        self._max_start = len(self.df) - self.H - 1

    # -- helpers ----------------------------------------------------------

    def _obs(self):
        f = self._feats_scaled[self.t]
        parts = [f]
        if self.use_signal:
            parts.append(np.array([self.signal.iloc[self.t]], dtype=np.float32))
        inv_frac = self.inventory / self.target_qty
        time_frac = (self.steps_left) / self.H
        parts.append(np.array([inv_frac, time_frac], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _limit_fill_prob(self):
        """Probability a passive limit sell at best ask fills this step.

        Higher trade intensity and thinner queue -> more likely to fill.
        Simple, monotone proxy calibrated to [0.05, 0.85].
        """
        inten = self.intensity[self.t]
        # squashing: busy market => higher fill chance
        p = 1.0 - np.exp(-inten / 120.0)
        return float(np.clip(p, 0.05, 0.85))

    def _market_impact(self, qty):
        """Temporary impact (fraction of mid): eta * (qty/depth)^alpha.

        Functional form and coefficient are UNCALIBRATED model choices;
        results are reported across a sweep of (eta, alpha) — see
        run_sensitivity.py — rather than defended at a single value.
        """
        d = max(self.depth[self.t], 1.0)
        return self.impact_eta * (qty / d) ** self.impact_alpha

    # -- gym API ----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.start = int(self.rng.integers(0, max(self._max_start, 1)))
        self.t = self.start
        self.steps_left = self.H
        self.inventory = self.target_qty
        self.arrival_mid = self.mid[self.t]      # benchmark price
        self.realized_value = 0.0                # cash from sells
        self.executed_qty = 0.0
        return self._obs(), {}

    def step(self, action, qty_override=None):
        """Advance one step.

        action: 0=hold, 1=limit sell, 2=market sell
        qty_override: if given, sell exactly this size this step (used by
            volume-aware baselines like VWAP/POV). If None, uses the default
            time-uniform slice. The Gym API only ever passes `action`.
        """
        mid = self.mid[self.t]
        half_spread = self.spread[self.t] / 2.0
        # default slice: time-uniform (target spread over the horizon)
        if qty_override is not None:
            slice_qty = min(self.inventory, max(qty_override, 0.0))
        else:
            slice_qty = min(self.inventory, self.target_qty / self.H * 1.5)
        reward = 0.0
        exec_px = None
        exec_qty = 0.0

        if action == 2 and self.inventory > 1e-9:        # MARKET sell
            exec_qty = slice_qty
            impact = self._market_impact(exec_qty) * mid
            exec_px = (mid - half_spread) - impact       # sell below mid
            fee = exec_px * self.fee_bps / 1e4
            exec_px -= fee

        elif action == 1 and self.inventory > 1e-9:      # LIMIT sell
            ask_px = mid + half_spread                    # post at best ask
            if self.tape is not None:
                # REAL fill attribution: fill only what real buy-aggressor
                # trades at/through our price would have taken this step.
                t0 = self.ts_ns[self.t]
                t1 = (self.ts_ns[self.t + 1]
                      if self.t + 1 < len(self.ts_ns) else t0 + 1_000_000_000)
                fillable = self.tape.sell_fillable(t0, t1, ask_px)
                if fillable > 1e-9:
                    exec_qty = min(slice_qty, fillable)
                    exec_px = ask_px                      # fill at our limit
            else:
                # fallback proxy (no tape provided) — flagged as approximate
                if self.rng.random() < self._limit_fill_prob():
                    exec_qty = slice_qty
                    exec_px = ask_px

        # action 0 (HOLD): nothing

        if exec_px is not None and exec_qty > 1e-9:
            self.realized_value += exec_px * exec_qty
            self.inventory -= exec_qty
            self.executed_qty += exec_qty
            # reward = negative shortfall for this slice (in bps of arrival)
            shortfall_bps = (self.arrival_mid - exec_px) / self.arrival_mid * 1e4
            reward = -shortfall_bps * (exec_qty / self.target_qty)

        # advance time
        self.t += 1
        self.steps_left -= 1

        terminated = False
        truncated = False
        if self.steps_left <= 0 or self.inventory <= 1e-9:
            terminated = True
            # force-liquidate any remainder at a punitive market price
            if self.inventory > 1e-9:
                mid_now = self.mid[self.t]
                impact = self._market_impact(self.inventory) * mid_now
                px = (mid_now - self.spread[self.t]/2.0) - impact
                self.realized_value += px * self.inventory
                shortfall_bps = (self.arrival_mid - px) / self.arrival_mid * 1e4
                # extra penalty for failing to work the order
                reward += -shortfall_bps * (self.inventory / self.target_qty) * 1.5
                self.executed_qty += self.inventory
                self.inventory = 0.0

        obs = self._obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32)
        info = {}
        if terminated:
            avg_px = self.realized_value / self.target_qty
            info["avg_exec_px"] = avg_px
            info["arrival_mid"] = self.arrival_mid
            info["shortfall_bps"] = (self.arrival_mid - avg_px) / self.arrival_mid * 1e4
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Classical baselines (for Layer 4 comparison) — run on the same env slices
# ---------------------------------------------------------------------------

def run_twap(env):
    """TWAP: sell an equal slice via market order every step."""
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    info = {}
    while not done:
        obs, r, term, trunc, info = env.step(2)  # always market a slice
        total_r += r
        done = term or trunc
    return total_r, info


def run_passive(env):
    """Naive passive: always post limit, hope it fills (forced liq at end)."""
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    info = {}
    while not done:
        obs, r, term, trunc, info = env.step(1)
        total_r += r
        done = term or trunc
    return total_r, info


def run_vwap(env):
    """VWAP: sell more when volume is high, less when thin.

    Slices are weighted by the trade-intensity profile over the episode
    window, so the schedule front/back-loads to match where the market
    actually trades. Each slice is sent as a market order.
    (Approximates the Barzykin & Lillo optimal-VWAP idea: track the
    volume curve rather than the clock.)
    """
    obs, _ = env.reset()
    # volume profile over this episode's window
    start, H = env.start, env.H
    vol = env.intensity[start:start + H].astype(np.float64)
    vol = np.where(vol > 0, vol, vol.mean() if vol.mean() > 0 else 1.0)
    weights = vol / vol.sum()
    target = env.target_qty

    done = False
    total_r = 0.0
    info = {}
    step_i = 0
    while not done:
        # this step's share of total qty, by volume weight
        slice_qty = target * weights[min(step_i, H - 1)]
        obs, r, term, trunc, info = env.step(2, qty_override=slice_qty)
        total_r += r
        step_i += 1
        done = term or trunc
    return total_r, info


def run_pov(env, participation=0.20):
    """POV (Percentage of Volume): trade a slice proportional to the share
    of total episode volume occurring this step, scaled by participation
    urgency, with a catch-up floor so the order finishes on time.

    The schedule tracks the volume *profile* (like VWAP) but is driven by
    a participation target rather than a precomputed weight curve.
    """
    obs, _ = env.reset()
    start, H = env.start, env.H
    vol = env.intensity[start:start + H].astype(np.float64)
    total_vol = vol.sum()
    if total_vol <= 0:
        vol = np.ones(H)
        total_vol = H

    done = False
    total_r = 0.0
    info = {}
    step_i = 0
    while not done:
        # fraction of the window's volume happening this step
        vol_share = vol[min(step_i, H - 1)] / total_vol
        # base slice follows the volume profile; participation tilts urgency
        slice_qty = env.target_qty * vol_share * (1.0 + participation)
        # catch-up floor: never fall so far behind we can't finish
        steps_left = max(env.steps_left, 1)
        min_needed = env.inventory / steps_left
        slice_qty = max(slice_qty, min_needed * 0.5)
        # cap: don't blow through more than a sane chunk at once
        slice_qty = min(slice_qty, env.inventory)
        obs, r, term, trunc, info = env.step(2, qty_override=slice_qty)
        total_r += r
        step_i += 1
        done = term or trunc
    return total_r, info