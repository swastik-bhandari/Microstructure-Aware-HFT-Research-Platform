# Microstructure-Aware HFT Research Platform

A real-data testbed for **optimal trade execution** on high-frequency limit-order-book
(LOB) data, built on the open Hyperliquid Level-4 dataset. The platform reconstructs
the order book from raw order-lifecycle events, engineers microstructure features,
trains a CNN-LSTM directional signal and a PPO execution agent, and benchmarks the
agent against classical execution strategies (TWAP / VWAP / POV) under a **real
trade-attribution fill model**.

> **Headline finding.** Under an idealized (probabilistic) fill model, the PPO agent
> appears to beat VWAP by **~19 bps**. Under a fill model grounded in the actual
> recorded trades, that advantage **collapses to zero** (a statistical tie). When the
> agent is then given the same volume-sizing flexibility VWAP enjoys, it recovers a
> **small but significant ~2 bps edge**. The project's primary contribution is
> therefore methodological: *reported RL execution advantages are highly sensitive to
> how passive fills are simulated.*

---

## Table of contents
1. [What this project is](#1-what-this-project-is)
2. [Data](#2-data)
3. [Architecture: the four layers](#3-architecture-the-four-layers)
4. [Methodology (formulas + process)](#4-methodology-formulas--process)
5. [Reproducing the pipeline](#5-reproducing-the-pipeline-end-to-end)
6. [Results](#6-results)
7. [Analysis of results](#7-analysis-of-results)
8. [Validation](#8-validation)
9. [Limitations & future work](#9-limitations--future-work)
10. [Repository map](#10-repository-map)

---

## 1. What this project is

The task is **single-asset optimal liquidation**: sell a fixed quantity of ETH
perpetual within a fixed time window, minimizing **implementation shortfall** (the
cost of executing relative to the price available when the order arrived).

The platform is organized into four layers:

| Layer | Component | Status |
|------|-----------|--------|
| 1 | LOB reconstruction + price-time-priority matching engine | ✅ Python reference, validated |
| 2 | Microstructure feature engine (6 features) | ✅ validated |
| 3a | CNN-LSTM short-horizon price-direction signal | ✅ trained (54.0% test acc) |
| 3b | PPO execution agent | ✅ trained, multi-seed |
| 4 | Backtesting vs TWAP/VWAP/POV + sensitivity analysis | ✅ complete |

All results are produced on **one trading day of real ETH data (2025-12-01)** and
are saved to disk as CSV/JSON evidence plus trained model checkpoints.

---

## 2. Data

**Source.** *An Open Book: Level 4 Order Book Data from the Hyperliquid Exchange*
(Albers et al., 2026), Zenodo DOI `10.5281/zenodo.18184441`, CC-BY-4.0.

Three streams exist in the dataset; this project uses two:

* **Order statuses** (`eth_orders_202512.tar.xz`) — the complete order lifecycle:
  every order arrival, cancellation, modification, fill, **and rejection**, in a
  custom 54-byte binary format. Rejected orders are data unavailable from almost any
  other source. Organized per-coin, per-hour (`YYYYMMDD/eth_HH.data.gz`).
* **Trades** (`trades_2025_12.tar`) — executed trades for all coins, gzip JSON lines,
  one file per hour (`YYYYMMDD/HH.gz`).
* **Lookup tables** (`mapdir.tar.xz`) — integer→label maps for order types, statuses,
  time-in-force; used by the reader to decode the binary records.

Only **2025-12-01** was extracted (the full dataset is ~195 GB). The official reader
`read_data.py` decodes the binary format (including the bit-packed price/size
encoding) and is used as-is — no custom binary parser was written.

**Key dataset facts used in the methodology:**
* Price/size fields are bit-packed: `decimals = encoded >> 29`,
  `value = encoded & 0x1FFFFFFF`, `price = value / 10^decimals`. (Handled by the reader.)
* Order side is the boolean `isAsk` (`True` = sell). Trade side is `"A"`/`"B"`
  (`"B"` = **buy-aggressor**, the side that can fill a resting *sell*).
* 18 distinct statuses (e.g. `open`, `filled`, `canceled`, and 7 rejection types).
* Only `Alo` (post-only) and `Gtc` time-in-force limit orders rest in the book;
  `Ioc`/market/trigger orders never rest.

---

## 3. Architecture: the four layers

```
 raw order events                        real trades
        |                                     |
        v                                     v
 ┌───────────────┐                    ┌────────────────┐
 │ Layer 1: LOB   │                    │ Trade tape     │
 │ matching engine│                    │ (.npz, sorted) │
 └──────┬─────────┘                    └───────┬────────┘
        │ book state every event               │ fill attribution
        v                                       │
 ┌───────────────┐                              │
 │ Layer 2:       │  6 features / 1s            │
 │ feature engine │─────────────┐               │
 └───────────────┘              v               │
                         ┌──────────────┐        │
                         │ Layer 3a:    │ P(up)  │
                         │ CNN-LSTM     │───┐     │
                         └──────────────┘   v     v
                                       ┌─────────────────┐
                                       │ Layer 3b: PPO   │
                                       │ execution agent │
                                       └────────┬────────┘
                                                v
                                       ┌─────────────────┐
                                       │ Layer 4:        │
                                       │ backtest vs     │
                                       │ TWAP/VWAP/POV   │
                                       └─────────────────┘
```

---

## 4. Methodology (formulas + process)

### 4.1 Layer 1 — LOB reconstruction with matching

The book is `{price: {oid: size}}` for each side; Python dicts are insertion-ordered,
so inner-dict order encodes **time priority**. Events are replayed in timestamp order
(`lob_engine.py`, class `LOB`):

* **open** (only `Limit` orders with TIF ∈ {`Alo`,`Gtc`}): first *match* against the
  opposite side, then rest any remainder.
* **filled**: reduce/remove the order. On a partial fill the event's `sz` is the
  *remaining* size (an empirical ambiguity resolved during validation — see §8).
* **canceled** (and 7 cancel variants): remove the order.
* **rejected** (7 variants): never touched the book — counted as a separate signal.

**Matching rule (price-time priority).** An incoming buy at price `p` consumes asks
with price ≤ `p`, cheapest level first, oldest order first within a level, until the
incoming size is exhausted or the book no longer crosses (mirror logic for sells):

```
while remaining > 0 and best_opposite crosses p:
    for resting_order in level (insertion order = time priority):
        traded = min(remaining, resting_order.size)
        remaining        -= traded
        resting_order.size -= traded
```

This matching step is what keeps the reconstructed book non-crossing (bid < ask),
which is the core correctness property validated in §8.

### 4.2 Layer 2 — feature engine

The book is updated on **every event**; a feature vector is sampled on a **fixed 1-second
grid** (`build_full_day.py`, sampling via `lob_engine.FeatureEngine`). Event-driven
updates preserve correctness; time-sliced sampling gives the RL agent and CNN-LSTM
evenly-spaced observations. Six features per state:

| Feature | Definition |
|---|---|
| `spread` | `best_ask − best_bid` |
| `obi` (imbalance) | `bid_vol / (bid_vol + ask_vol)` over the top 5 levels, ∈ [0,1] |
| `depth` | total resting volume over the top 5 levels, both sides |
| `volatility` | rolling std of mid-price **log returns** over a 20-sample window |
| `trade_intensity` | EWMA of fills-per-second, half-life 5 s |
| `queue_at_best` | resting size at the best bid (FIFO queue-position proxy) |

`trade_intensity` EWMA update, for elapsed `dt` seconds and `new_fills` since the last
sample:

```
rate  = new_fills / dt
alpha = 1 − exp(−dt / halflife)          # halflife = 5 s
intensity = (1 − alpha)·intensity + alpha·rate
```

`volatility` = `std( diff( log(mid_history) ) )` over the trailing window.

Output: `states_eth_20251201.csv`, **86,399** one-second states for the full day.

### 4.3 Layer 3a — CNN-LSTM directional signal

Predicts the **direction of the mid-price over the next 30 s**, producing `P(up)` as a
candidate 7th feature for the agent (`train_cnn_lstm_v2.py`).

**Inputs (10 features).** The 6 base features plus 4 engineered *signed* features
(the base set is mostly magnitude-only, which carries little directional information):

```
mid_return  = pct_change(mid)
mid_accel   = diff(mid_return)
obi_trend   = rolling_mean(obi − 0.5, window 10)
signed_flow = (obi − 0.5) · trade_intensity
```

**Labels (binary, with a meaningful-move threshold).** For horizon `H = 30` steps:

```
fwd_ret   = (mid[t+H] − mid[t]) / mid[t]
roll_vol  = rolling_std(pct_change(mid), 60)
threshold = move_mult · roll_vol          # move_mult = 1.5
label = 1   if fwd_ret >  threshold       (up)
label = 0   if fwd_ret < −threshold       (down)
label = −1  otherwise                     (chop — EXCLUDED from training)
```

Excluding chop (rather than forcing a 3rd class) makes the target learnable: the model
is trained only to separate decisive up-moves from decisive down-moves.

**Sequence model.** Sliding windows of `seq_len = 60` states (the last 60 seconds):

```
Conv1d(10→48, k3) → ReLU → Conv1d(48→48, k3) → ReLU
   → LSTM(48→96) → take last timestep
   → Linear(96→48) → ReLU → Dropout(0.3) → Linear(48→2) → softmax
```

**Split.** Chronological 70/15/15 (train/val/test) — *no shuffling*, to avoid leaking
future information into the past. Features z-scored on train statistics only
(`scaler_v2.joblib`).

### 4.4 Layer 3b — PPO execution environment

A Gymnasium environment (`execution_env.py`, class `ExecutionEnv`) replays the state
table. Episode: sell `target_qty = 10` ETH over `horizon = 60` one-second steps.

* **Observation:** the 6 (or 7, with the signal) z-scored features + `[inventory_frac,
  time_frac]`.
* **Actions (baseline env, v1):** `0 = hold`, `1 = limit sell at best ask`,
  `2 = market sell`. Fixed slice size = `target_qty / H · 1.5`.
* **Reward = negative implementation shortfall** for the slice filled this step, in bps
  of the arrival mid, weighted by the fraction of the order it represents:

```
shortfall_bps = (arrival_mid − exec_px) / arrival_mid · 1e4
reward        = − shortfall_bps · (exec_qty / target_qty)
```

* **Terminal force-liquidation.** Any unsold inventory at the horizon is liquidated at
  a market price with an extra ×1.5 penalty, so the agent cannot "win" by simply never
  selling.

**Temporary market-impact model** (applied to market orders and the terminal liquidation):

```
impact_fraction = eta · (qty / depth)^alpha          # of mid price
exec_px(market) = (mid − spread/2) − impact·mid − fee
```

`(eta, alpha)` are **uncalibrated modeling choices**; rather than defend one value, the
result is reported across a sweep (§6.3). Default `(eta, alpha) = (0.5, 0.5)`; taker
`fee = 2 bps`.

#### The fill model — the crux of the project

A passive limit sell only earns a good price *if it actually fills*. Two models are
implemented and **directly compared**:

* **Idealized (proxy) fill model** — a probability increasing with trade intensity:

  ```
  p_fill = clip( 1 − exp(−trade_intensity / 120), 0.05, 0.85 )
  ```

  This is an invented functional form (the constant 120 is not calibrated).

* **Real trade-attribution fill model** (`TradeTape`, default) — grounded in the
  recorded trades. A resting sell at price `P` over the interval `[t0, t1)` fills only
  up to the volume of **buy-aggressor** trades (`side == "B"`) at price ≥ `P` in that
  interval:

  ```
  fillable = Σ sz[i]   for trades i in [t0,t1) with isB[i] and px[i] ≥ P
  exec_qty = min(slice_qty, fillable)
  exec_px  = P
  ```

  i.e. the agent fills only if and to the extent a real buyer crossed its price. This
  replaces the invented probability with measured market behaviour. (Level-1
  attribution: price-based, no queue position — see §9.)

### 4.5 Layer 4 — baselines

All baselines run inside the *same* environment (identical fills, impact, reward), so
the comparison is apples-to-apples. They differ only in their decision rule:

| Strategy | Rule |
|---|---|
| **TWAP** | sell an equal slice `qty/N` every step (blind to the market) |
| **VWAP** | slice ∝ the trade-intensity (volume) profile over the window |
| **POV** | trade a fixed participation fraction of current volume, with a catch-up floor |
| **Passive** | always post a limit order, hope it fills (force-liquidate remainder) |

VWAP/POV size each slice by volume via a `qty_override` path in `step()`; PPO (v1) uses
a fixed slice — the asymmetry that motivates the fairness check (§6.4).

### 4.6 Statistical protocol

* Every PPO result is the **mean ± std over 5 random seeds** (1000–1004); 100,000
  training timesteps per agent; 300 evaluation episodes on the held-out 30% slice.
* The CNN-LSTM contribution is tested as `PPO(6) − PPO(7)` across seeds; it is declared
  significant only if `|mean| ≥ 2 × standard_error`.
* All numbers, configs, seeds, and trained models are persisted (CSV + JSON + model
  zips) so any table can be regenerated.

---

## 5. Reproducing the pipeline (end-to-end)

Environment: Python 3.11/3.12 (3.14 has dependency issues), `torch`,
`stable-baselines3`, `gymnasium`, `scikit-learn`, `pandas`, `numpy`, `joblib`.
Data layout expected: `order_statuses/YYYYMMDD/eth_HH.data.gz`,
`trades/YYYYMMDD/HH.gz`, `mapdir/*.csv` (see §2).

```bash
# Layer 2: replay all 24 hours into the 1-second state table
python build_full_day.py                      # -> states_eth_20251201.csv
python analyze_states.py                       # -> overview figure + stats

# Trade tape for real fills
python preprocess_trades.py                    # -> trade_tape_eth_20251201.npz

# Layer 3a: train the CNN-LSTM directional signal
python train_cnn_lstm_v2.py                    # -> cnn_lstm_v2.pt, scaler_v2.joblib

# Layer 3b/4: the two-fill-model comparison (5 seeds each)
python run_variance.py --runs 5 --timesteps 100000 --tape none   # IDEALIZED
#   -> rename outputs to *_IDEALIZED.* (see note below)
python run_variance.py --runs 5 --timesteps 100000               # REAL fills
#   -> rename outputs to *_REALFILLS.*

# Impact-model sensitivity sweep (9 settings)
python run_sensitivity.py                      # -> sensitivity_results.csv

# Outlier re-test (one impact cell, more seeds/steps)
python retest_cell.py --alpha 0.5 --eta 0.5 --seeds 5 --timesteps 120000

# Fairness check: size-aware action space, real fills
python run_fairness.py --runs 5 --timesteps 100000   # -> fairness_summary.json
```

> **Note on filenames.** `run_variance.py` writes fixed names
> (`variance_results.csv`, `variance_summary.json`). Copy them to
> `*_IDEALIZED.*` / `*_REALFILLS.*` after each run so they are not overwritten. Both
> versioned copies are included in this repo.

---

## 6. Results

All shortfalls in **basis points (bps), lower is better**. Day: ETH 2025-12-01.

### 6.1 Headline: the same agent under two fill models

| Strategy | Idealized fills | Real-trade-attribution fills |
|---|---:|---:|
| **PPO (6 feat)** | **22.44 ± 1.87** | **41.63 ± 1.58** |
| PPO (7 feat, +P(up)) | 22.60 ± 0.37 | 41.75 ± 2.16 |
| VWAP | 41.71 | 41.71 |
| TWAP | 42.81 | 42.81 |
| POV | 46.07 | 46.07 |
| Passive | 53.18 | **167.70** |
| **PPO edge over VWAP** | **+19.27** | **+0.08** |
| CNN-LSTM contribution | −0.16 ± 0.92 (n.s.) | −0.12 ± 0.66 (n.s.) |

*(VWAP/TWAP/POV are deterministic schedules and are identical across fill models in the
v1 action space; only fill-dependent strategies — PPO and Passive — move.)*

### 6.2 CNN-LSTM directional model

* Binary up/down, 30 s horizon. **Test accuracy 54.0%** vs majority-class baseline
  51.7% and random 50.0% (`cnn_lstm_v2_meta.json`). A weak-but-real signal.

### 6.3 Impact-model sensitivity (real fills, PPO vs VWAP gap, bps)

| α \ η | 0.1 | 0.5 | 2.0 |
|---|---:|---:|---:|
| **0.5** | +0.06 | −25.71* | −2.56 |
| **0.6** | −0.00 | −0.09 | +2.08 |
| **1.0** | +0.04 | +0.01 | +0.04 |

`*` outlier — a short-training convergence failure (see §6.5). Across the other 8
settings the gap is tightly centered on **≈ 0** (median +0.01 bps): PPO ties VWAP
regardless of impact assumptions.

### 6.4 Fairness check — size-aware action space (real fills, 5 seeds)

Expanded action space: `hold` + `{limit, market} × {small, medium, large}` (7 actions),
so PPO can size slices like VWAP.

| Strategy | Shortfall (bps) |
|---|---:|
| **PPO size-aware (7 feat)** | **39.53 ± 1.31** |
| PPO size-aware (6 feat) | 43.42 ± 4.98 |
| VWAP | 41.71 |
| **Best PPO edge over VWAP** | **+2.18 (SE 0.65)** |

`+2.18 / 0.65 ≈ 3.4` standard errors → **statistically significant**. (TWAP/Passive in
this env are distorted by the size-aware action mapping and should be read from §6.1, not here — see §9.)

### 6.5 Outlier re-test

Re-running the `(α=0.5, η=0.5)` cell at 5 seeds × 120k steps:
**PPO 41.00 ± 1.35 vs VWAP 41.33 → +0.33 bps tie**
(`retest_alpha0.5_eta0.5_summary.json`, verdict `tie_converged_artifact`). The original
−25.71 was a single under-trained run, not a real effect.

---

## 7. Analysis of results

**1. The headline edge is a fill-model artifact.** PPO's +19 bps advantage over VWAP
under the idealized model **vanishes (+0.08 bps) under real fills**. The mechanism is
visible in the Passive baseline, which degrades from 53 → 168 bps: the idealized model
let passive limit orders fill far more readily — and at better prices — than real
buy-aggressor flow actually permits. The agent under the idealized model was largely
exploiting that generosity, not genuine alpha.

**2. The CNN-LSTM signal does not help execution.** Its contribution is statistically
zero under **both** fill models (−0.16 ± 0.92 and −0.12 ± 0.66). This is consistent with
its modest 54% directional accuracy: a weak predictor produces a weak (here,
indistinguishable-from-noise) execution effect. Direction prediction ≠ execution edge —
most of any execution quality comes from *timing and order-type choice*, not from
forecasting the next move.

**3. A fair action space recovers a small, real edge.** When PPO is allowed to size
slices (the one capability VWAP had and v1 PPO lacked), it beats VWAP by **+2.18 bps at
3.4σ** under real fills. The binding constraint on v1 was the *action space*, not the
learning. The honest conclusion is therefore nuanced: *idealized fills massively inflate
the apparent edge (≈19 bps); under real fills with a matched action space, a genuine but
much smaller edge (≈2 bps) remains.*

**4. Caution on the size-aware 6-vs-7 gap.** In §6.4 the 6-feature agent (43.4 ± 4.98)
looks worse than the 7-feature (39.5 ± 1.31), but the 6-feature std is large (two
unstable seeds). This reflects **training instability** under the larger action space,
not a resurrection of the CNN-LSTM signal. The clean, low-variance CNN-LSTM verdict is
the one in §6.1/§6.2: **not significant.**

---

## 8. Validation

The reconstruction and pipeline were validated at each layer, not assumed correct:

* **LOB correctness (Layer 1).** After implementing price-time matching, the
  reconstructed book is **0.0% crossed** (best_bid < best_ask in every snapshot), spread
  is pinned at ≈ $0.10 (one ETH tick), and the reconstructed **mid (2796–2838)** overlaps
  the **independently-recorded trade price range (2791–2838)** for the same hour. The
  reconstruction was iteratively debugged through three failure modes (stale orders at
  extreme prices → filtering non-resting TIFs → adding the matching step) until these
  invariants held.
* **Fill-event semantics.** The dataset's `filled` events were empirically ambiguous
  (remaining vs filled size). The interpretation "`sz` = remaining size" was selected
  because it is the one that keeps the book non-crossing and price-consistent with the
  trades — i.e. validated by its downstream effect rather than assumed.
* **Feature sanity (Layer 2).** All features pass invariants: `spread > 0`,
  `obi ∈ [0,1]`, `depth ≥ 0`, `volatility ≥ 0` across all 86,399 states; feature
  distributions are economically sensible (e.g. depth inversely related to volatility).
* **Trade tape.** 402,483 ETH trades for the day, 48.0% buy-aggressor, price range
  2718–2998 — consistent with the reconstructed book.
* **Statistical honesty.** Multi-seed variance (not single runs), an explicit
  significance test for the CNN-LSTM, a sensitivity sweep instead of a single
  hand-picked impact coefficient, and a re-test that resolved the one outlier. Every
  reported number is backed by a saved CSV/JSON and the trained model checkpoints.

---

## 9. Limitations & future work

* **Level-1 fill attribution only.** Fills are price-based: a resting order fills if
  *any* real trade crossed its price, ignoring **queue position**. Real fills would be
  somewhat worse (orders ahead in the FIFO queue fill first). The dataset's per-trade
  `oid` fields support **Level-2 (queue-aware) attribution** — the natural next step.
* **One day, one asset.** All results are 2025-12-01 ETH. The +2.18 bps fair-action
  edge is suggestive, not yet shown to be robust across regimes/assets; additional days
  can be extracted from the same archives.
* **Uncalibrated impact model.** Handled by a sensitivity sweep rather than a single
  calibrated coefficient; a literature-calibrated impact function would strengthen the
  absolute (not relative) numbers.
* **TWAP/Passive in the size-aware env (§6.4)** are distorted by the action mapping and
  are reported from the v1 env (§6.1); only the PPO-vs-VWAP comparison is valid in the
  size-aware env.
* **C++ engine.** The matching engine is a validated Python reference; a
  latency-optimized C++ port (for true latency/throughput benchmarking) remains
  engineering future work.

---

## 10. Repository map

**Pipeline scripts**
* `read_data.py` — official Hyperliquid binary/JSON reader (decodes records, mapdir).
* `lob_engine.py` — Layer 1 (`LOB`) + Layer 2 (`FeatureEngine`) + state builder.
* `build_full_day.py` — replays 24 h into `states_eth_20251201.csv`.
* `analyze_states.py` — day overview figure + descriptive stats.
* `preprocess_trades.py` — builds `trade_tape_eth_20251201.npz` for real fills.
* `train_cnn_lstm_v2.py` — Layer 3a CNN-LSTM directional model.
* `execution_env.py` — Layer 3b env (3-action) + TWAP/VWAP/POV/Passive baselines.
* `execution_env_v2.py` — size-aware (7-action) env for the fairness check.
* `run_variance.py` — multi-seed PPO vs baselines (`--tape none` = idealized).
* `run_sensitivity.py` — impact-model `(α, η)` sweep.
* `retest_cell.py` — re-test a single impact cell with more seeds/steps.
* `run_fairness.py` — size-aware PPO vs baselines under real fills.

**Saved evidence**
* `states_eth_20251201.csv` (86,399 states), `states_eth_20251201_overview.png`
* `trade_tape_eth_20251201.npz` (402,483 trades)
* `cnn_lstm_v2.pt`, `scaler_v2.joblib`, `cnn_lstm_v2_meta.json`
* `variance_summary_IDEALIZED.json`, `variance_summary_REALFILLS.json` (+ `.csv`)
* `sensitivity_results.csv`
* `retest_alpha0.5_eta0.5_summary.json` (+ `.csv`)
* `fairness_summary.json`, `fairness_results.csv`
* `models/ppo{6,7}_seed{1000..1004}.zip` — fixed-action agents
* `models/ppo{6,7}_sizeaware_seed{1000..1004}.zip` — size-aware agents

**Exploratory / development** (kept for provenance)
* `lob_reconstruct.py`, `cleaning_data.py`, `match_on_insertion.py`, `diagnostic.py`
  — the staged debugging scripts that produced the validated reconstruction.

---

### Citation

Dataset: Albers, J., Cucuringu, M., Howison, S., & Shestopaloff, A. Y. (2026).
*An Open Book: Level 4 Order Book Data from the Hyperliquid Exchange.* Zenodo.
https://doi.org/10.5281/zenodo.18184441 (CC-BY-4.0).