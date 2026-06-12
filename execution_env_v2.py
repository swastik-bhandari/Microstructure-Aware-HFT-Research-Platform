"""
HFT Research Platform — Execution Environment v2 (FAIRNESS CHECK)
=================================================================

Identical to execution_env.py EXCEPT the agent's action space is expanded
so PPO has the same slice-sizing flexibility that VWAP/POV enjoy.

Motivation (fairness check):
    In v1 the agent had 3 actions (hold/limit/market) and a FIXED slice size,
    while VWAP/POV size each slice by volume via qty_override. That asymmetry
    confounds "learning doesn't help" with "the agent was handicapped".

    v2 gives the agent 7 actions:
        0 = HOLD
        1 = LIMIT small      2 = LIMIT medium    3 = LIMIT large
        4 = MARKET small     5 = MARKET medium   6 = MARKET large
    where small/medium/large are fractions of the remaining-inventory pace.
    Now PPO can vary BOTH order type and size, plus react to all 6-7 features
    VWAP ignores. If it still ties VWAP under real fills, the null is robust.

Everything else (fill attribution, impact model, reward, baselines) is
imported unchanged from execution_env.py so the comparison stays apples-to-
apples. Only the agent's choice set differs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# reuse all the validated machinery from v1
from execution_env import (
    TradeTape, MARKET_FEATURES,
    run_twap, run_vwap, run_pov, run_passive,   # baselines unchanged
)


# slice-size multipliers applied to the time-uniform pace (target/H)
# small=0.5x, medium=1.5x, large=3.0x of the per-step pace
SIZE_MULT = {"small": 0.5, "medium": 1.5, "large": 3.0}


class ExecutionEnvV2(gym.Env):
    """Sell target_qty ETH over horizon_steps, with size-aware actions.

    Reuses the exact fill / impact / reward logic of ExecutionEnv but exposes
    7 discrete actions so the policy controls slice SIZE as well as type.
    """

    metadata = {"render_modes": []}

    # 7 actions: hold + {limit,market} x {small,medium,large}
    _ACTIONS = [
        ("hold", None),
        ("limit", "small"), ("limit", "medium"), ("limit", "large"),
        ("market", "small"), ("market", "medium"), ("market", "large"),
    ]

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
        self.impact_eta = impact_eta
        self.impact_alpha = impact_alpha
        self.fee_bps = fee_bps
        self.tape = trade_tape
        self.rng = np.random.default_rng(seed)

        self.ts_ns = self.df["ts"].astype("int64").to_numpy()

        feats = self.df[MARKET_FEATURES].to_numpy(dtype=np.float32)
        self.f_mean = feats.mean(0)
        self.f_std = feats.std(0) + 1e-8
        self._feats_scaled = (feats - self.f_mean) / self.f_std

        self.mid = self.df["mid"].to_numpy(dtype=np.float64)
        self.spread = self.df["spread"].to_numpy(dtype=np.float64)
        self.depth = self.df["depth"].to_numpy(dtype=np.float64)
        self.intensity = self.df["trade_intensity"].to_numpy(dtype=np.float64)

        n_market = len(MARKET_FEATURES) + (1 if self.use_signal else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_market + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self._ACTIONS))  # 7

        self._max_start = len(self.df) - self.H - 1

    # -- helpers (identical to v1) ---------------------------------------

    def _obs(self):
        f = self._feats_scaled[self.t]
        parts = [f]
        if self.use_signal:
            parts.append(np.array([self.signal.iloc[self.t]], dtype=np.float32))
        inv_frac = self.inventory / self.target_qty
        time_frac = self.steps_left / self.H
        parts.append(np.array([inv_frac, time_frac], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _market_impact(self, qty):
        d = max(self.depth[self.t], 1.0)
        return self.impact_eta * (qty / d) ** self.impact_alpha

    def _slice_for(self, size_key):
        """Slice size from the chosen bucket, capped at remaining inventory."""
        pace = self.target_qty / self.H          # time-uniform per-step pace
        q = pace * SIZE_MULT[size_key]
        return float(min(self.inventory, max(q, 0.0)))

    # -- gym API ----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.start = int(self.rng.integers(0, max(self._max_start, 1)))
        self.t = self.start
        self.steps_left = self.H
        self.inventory = self.target_qty
        self.arrival_mid = self.mid[self.t]
        self.realized_value = 0.0
        self.executed_qty = 0.0
        return self._obs(), {}

    def step(self, action, qty_override=None):
        kind, size_key = self._ACTIONS[int(action)]
        # baselines (VWAP/POV/TWAP) drive size via qty_override and
        # pass action=2 (=market,small); when qty_override is given we
        # treat the action purely as "market" with that exact size.
        if qty_override is not None:
            kind = "market"
        mid = self.mid[self.t]
        half_spread = self.spread[self.t] / 2.0
        reward = 0.0
        exec_px = None
        exec_qty = 0.0

        if qty_override is not None:
            slice_qty = float(min(self.inventory, max(qty_override, 0.0)))
        else:
            slice_qty = 0.0 if kind == "hold" else self._slice_for(size_key)

        if kind == "market" and self.inventory > 1e-9:
            exec_qty = slice_qty
            impact = self._market_impact(exec_qty) * mid
            exec_px = (mid - half_spread) - impact
            fee = exec_px * self.fee_bps / 1e4
            exec_px -= fee

        elif kind == "limit" and self.inventory > 1e-9:
            ask_px = mid + half_spread
            if self.tape is not None:
                t0 = self.ts_ns[self.t]
                t1 = (self.ts_ns[self.t + 1]
                      if self.t + 1 < len(self.ts_ns) else t0 + 1_000_000_000)
                fillable = self.tape.sell_fillable(t0, t1, ask_px)
                if fillable > 1e-9:
                    exec_qty = min(slice_qty, fillable)
                    exec_px = ask_px
            else:
                inten = self.intensity[self.t]
                p = float(np.clip(1.0 - np.exp(-inten / 120.0), 0.05, 0.85))
                if self.rng.random() < p:
                    exec_qty = slice_qty
                    exec_px = ask_px

        if exec_px is not None and exec_qty > 1e-9:
            self.realized_value += exec_px * exec_qty
            self.inventory -= exec_qty
            self.executed_qty += exec_qty
            shortfall_bps = (self.arrival_mid - exec_px) / self.arrival_mid * 1e4
            reward = -shortfall_bps * (exec_qty / self.target_qty)

        self.t += 1
        self.steps_left -= 1

        terminated = False
        truncated = False
        if self.steps_left <= 0 or self.inventory <= 1e-9:
            terminated = True
            if self.inventory > 1e-9:
                mid_now = self.mid[self.t]
                impact = self._market_impact(self.inventory) * mid_now
                px = (mid_now - self.spread[self.t]/2.0) - impact
                self.realized_value += px * self.inventory
                shortfall_bps = (self.arrival_mid - px) / self.arrival_mid * 1e4
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