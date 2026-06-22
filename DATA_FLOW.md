# Data Flow & Dataset Metadata

A complete map of **what every file contains** and the **input → output of each
pipeline step**, so the project is fully reproducible and auditable.

For each artifact below you get: the schema (columns/fields, types, units), how to
inspect it yourself, and which step produces and consumes it.

---

## 0. How to inspect any file yourself

```python
import pandas as pd, numpy as np, json, joblib

# CSV (states, results)
df = pd.read_csv("states_eth_20251201.csv")
print(df.shape); print(df.dtypes); print(df.head())

# NPZ (trade tape)
z = np.load("trade_tape_eth_20251201.npz")
print(z.files); print({k: z[k].shape for k in z.files})

# JSON (summaries, metadata)
print(json.load(open("variance_summary_REALFILLS.json")))

# Scaler (fitted normalizer)
s = joblib.load("scaler_v2.joblib"); print(s.mean_, s.scale_)

# PyTorch model
import torch; sd = torch.load("cnn_lstm_v2.pt", map_location="cpu")
print(list(sd.keys())[:5])
```

For the raw archives (before extraction):
```bash
tar -tf eth_orders_202512.tar.xz | head     # list contents, no extract
tar -tf trades_2025_12.tar | head
python -c "from read_data import read_orders; print(read_orders('order_statuses','mapdir',date='2025-12-01',hour=12,coin='eth').head())"
```

---

## 1. RAW INPUTS (from Zenodo, before any processing)

### 1a. Order statuses — `order_statuses/YYYYMMDD/eth_HH.data.gz`
Binary, **54 bytes/record**, little-endian packed C struct. One file per hour.
Decoded by `read_data.read_orders(...)` into a DataFrame with these fields:

| Field | Type | Size | Units / meaning |
|---|---|---|---|
| `ts` | uint64 | 8 B | event timestamp, **nanoseconds** since Unix epoch |
| `userId` | uint32 | 4 B | pseudonymous trader id (→ `users.csv`) |
| `isBuilder` | bool | 1 B | builder field present |
| `statusId` | uint8 | 1 B | lifecycle status (→ `statuses.csv`); 18 values |
| `isAsk` | bool | 1 B | **true = sell**, false = buy |
| `limitPx` | uint32 | 4 B | limit price (**bit-packed**, decoded to float) |
| `sz` | uint32 | 4 B | **remaining** size at this event (bit-packed) |
| `oid` | uint64 | 8 B | unique order id — tracks an order across its lifecycle |
| `timestampDiff` | uint32 | 4 B | ms since the order was submitted |
| `tifId` | uint8 | 1 B | time-in-force (→ `tifs.csv`): Alo/Gtc/Ioc/… |
| `triggerPx` | uint32 | 4 B | trigger price (bit-packed) |
| `origSz` | uint32 | 4 B | size at submission (bit-packed) |

**Price/size decoding** (done by the reader): `decimals = enc >> 29`,
`value = enc & 0x1FFFFFFF`, `price = value / 10^decimals`.
**Lifecycle:** one `oid` appears multiple times — e.g. `open` → `filled` (with
*decreasing* `sz`) → `canceled`. Rejected orders appear once with a `*Rejected` status.

### 1b. Trades — `trades/YYYYMMDD/HH.gz`
Gzip JSON lines, **all coins mixed**, one file per hour. Fields used:

| Field | Type | Meaning |
|---|---|---|
| `time` | timestamp | trade time (→ ns) |
| `coin` | string | filter to `"ETH"` |
| `side` | string | `"A"` = sell-aggressor, **`"B"` = buy-aggressor** |
| `px` | float | execution price |
| `sz` | float | execution size |
| `hash` | string | L1 tx hash (dedup; `0x000…` for same-block trades) |
| `side_info[].oid` | uint64 | order ids of *both* counterparties (enables Level-2 attribution) |

### 1c. Lookup tables — `mapdir/*.csv`
`order_types.csv` (7 types), `statuses.csv` (18 statuses, incl. 7 rejection types),
`tifs.csv` (Alo=0, Gtc=1, Ioc=2, …), `users.csv` (id→address), `value_stats.csv`
(per-user record counts; not used in the pipeline). Used by the reader to decode
integer id fields to labels.

---

## 2. STEP-BY-STEP: inputs → outputs

### STEP 1 — Build the state table (Layer 1 + 2)
**Script:** `build_full_day.py` (uses `lob_engine.py`)

| | |
|---|---|
| **Input** | `order_statuses/20251201/eth_00..23.data.gz` (24 files), `mapdir/` |
| **Process** | Replay all order events in ts order through one continuous `LOB` (price-time matching, carry book across hour boundaries). Sample the 6 features on a fixed **1-second grid**. |
| **Output** | `states_eth_20251201.csv` |

**Output schema** — `states_eth_20251201.csv`, **shape (86 399, 9)**:

| Column | Type | Units | Definition |
|---|---|---|---|
| `ts` | int64 | ns | sample timestamp (1 s apart) |
| `mid` | float64 | USD | `(best_bid + best_ask)/2` |
| `spread` | float64 | USD | `best_ask − best_bid` |
| `obi` | float64 | [0,1] | bid_vol / (bid_vol+ask_vol), top 5 levels |
| `depth` | float64 | ETH | total resting vol, top 5 levels both sides |
| `volatility` | float64 | — | rolling std of mid log-returns (20-sample) |
| `trade_intensity` | float64 | fills/s | EWMA of fills, 5 s half-life |
| `queue_at_best` | float64 | ETH | resting size at best bid (queue proxy) |
| `hour` | int64 | 0–23 | UTC hour (provenance) |

`ts` range: `1764547200867476878` … `1764633598867476878` (full day, 2025-12-01 UTC).

### STEP 2 — Build the trade tape (for real fills)
**Script:** `preprocess_trades.py`

| | |
|---|---|
| **Input** | `trades/20251201/00..23.gz` (24 files), filtered to `coin == "ETH"` |
| **Process** | Concatenate, convert `time`→ns int64, mark `isB = (side=="B")`, **sort by ts** |
| **Output** | `trade_tape_eth_20251201.npz` |

**Output schema** — `trade_tape_eth_20251201.npz`, 4 parallel arrays, **402 483 trades**:

| Array | dtype | shape | meaning |
|---|---|---|---|
| `ts_ns` | int64 | (402483,) | trade timestamp, ns, sorted |
| `px` | float32 | (402483,) | execution price |
| `sz` | float32 | (402483,) | execution size |
| `isB` | bool | (402483,) | buy-aggressor (can fill a resting **sell**) |

Sanity: 48.0% buy-aggressor; px range 2718–2998 (matches the reconstructed book).

### STEP 3 — Train the CNN-LSTM direction signal (Layer 3a)
**Script:** `train_cnn_lstm_v2.py`

| | |
|---|---|
| **Input** | `states_eth_20251201.csv` |
| **Process** | Engineer 4 signed features → 10 total; binary up/down labels (chop excluded); z-score (train-only); 60-step windows; train Conv1d→LSTM→softmax; chronological 70/15/15 split |
| **Output** | `cnn_lstm_v2.pt`, `scaler_v2.joblib`, `cnn_lstm_v2_meta.json` |

**Model input tensor:** `(batch, seq_len=60, n_features=10)`.
**10 features (order matters — this is the scaler/model contract):**
`spread, obi, depth, volatility, trade_intensity, queue_at_best, mid_return,
mid_accel, obi_trend, signed_flow`.
**Model output:** 2 logits → softmax → `P(down), P(up)`. `P(up)` is the 7th agent feature.

`scaler_v2.joblib` (StandardScaler, `n_features_in_=10`) — stored `mean_`/`scale_` per
feature; **must** be applied before the model, in the exact feature order above.
`cnn_lstm_v2_meta.json`: `{horizon_s:30, seq_len:60, move_mult:1.5, features:[…10…],
test_acc:0.5397, majority:0.5169}`.

### STEP 4 — PPO training + baseline comparison (Layer 3b + 4)
**Script:** `run_variance.py` (env: `execution_env.py`)

| | |
|---|---|
| **Input** | `states_eth_20251201.csv`, `cnn_lstm_v2.pt` + `scaler_v2.joblib` (for P(up)), `trade_tape_eth_20251201.npz` (omit with `--tape none` for idealized fills) |
| **Process** | Compute P(up) per state; train PPO (6-feat & 7-feat) × 5 seeds; evaluate vs TWAP/VWAP/POV/Passive on held-out 30%; significance test |
| **Output** | `variance_results.csv`, `variance_summary.json`, `models/ppo{6,7}_seed{1000..1004}.zip` |

**Env observation vector** (per step): `n_market` z-scored features (6 or 7) +
`[inventory_frac, time_frac]`. **Actions:** `0=hold, 1=limit@ask, 2=market`.
**Reward:** `−shortfall_bps · (exec_qty/target_qty)` (see methodology).

**`variance_results.csv`** — per-seed: `seed, ppo6_bps, ppo7_bps` (5 rows).
**`variance_summary.json`** — `{fill_model, config{runs,timesteps,target_qty,horizon,
eval_episodes,seeds}, results{ppo6_mean_bps, ppo6_std_bps, ppo7_*, vwap_bps, twap_bps,
pov_bps, passive_bps, cnn_lstm_contribution_mean_bps, …_stderr_bps,
ppo_vs_best_classical_bps}}`. Two versioned copies exist:
`*_IDEALIZED.*` (`--tape none`) and `*_REALFILLS.*` (real tape).

### STEP 5 — Impact sensitivity sweep
**Script:** `run_sensitivity.py`

| | |
|---|---|
| **Input** | states CSV + tape |
| **Process** | Train PPO(6) + eval VWAP/TWAP at each `(alpha, eta)` in {0.5,0.6,1.0}×{0.1,0.5,2.0} |
| **Output** | `sensitivity_results.csv` |

**Schema:** `alpha, eta, ppo6_bps, vwap_bps, twap_bps, ppo_vs_vwap_gap_bps` (9 rows).

### STEP 6 — Outlier re-test
**Script:** `retest_cell.py --alpha 0.5 --eta 0.5 --seeds 5 --timesteps 120000`

| | |
|---|---|
| **Output** | `retest_alpha0.5_eta0.5.csv` (`seed, ppo_bps`), `retest_alpha0.5_eta0.5_summary.json` (`ppo_mean_bps, ppo_std_bps, vwap_bps, gap_vwap_minus_ppo_bps, stderr_bps, verdict, original_sweep_gap_bps`) |

### STEP 7 — Fairness check (size-aware actions)
**Script:** `run_fairness.py` (env: `execution_env_v2.py`)

| | |
|---|---|
| **Input** | states CSV + tape + CNN-LSTM |
| **Process** | Same as Step 4 but **7 actions** (`hold + {limit,market}×{small,medium,large}`), real fills, 5 seeds |
| **Output** | `fairness_results.csv` (`seed, ppo6_sizeaware_bps, ppo7_sizeaware_bps`), `fairness_summary.json`, `models/ppo{6,7}_sizeaware_seed{1000..1004}.zip` |

`fairness_summary.json.results`: `ppo6_sizeaware_mean_bps, ppo7_sizeaware_mean_bps,
vwap_bps, twap_bps, pov_bps, passive_bps, best_ppo_vs_vwap_bps, stderr_bps, ties_vwap`.

---

## 3. Dependency graph (what feeds what)

```
raw orders ─► build_full_day.py ─► states_eth_20251201.csv ─┬─► train_cnn_lstm_v2.py ─► cnn_lstm_v2.pt
                                                            │                            scaler_v2.joblib
raw trades ─► preprocess_trades.py ─► trade_tape...npz ──┐  │                            cnn_lstm_v2_meta.json
                                                         │  │
                                                         ▼  ▼  ▼ (P(up))
                                          run_variance.py / run_sensitivity.py /
                                          retest_cell.py / run_fairness.py
                                                         │
                                                         ▼
                                    variance/sensitivity/fairness *.csv + *.json
                                    + models/*.zip
```

**Contracts to respect when reproducing:**
1. Feature order in §STEP 3 is fixed — the scaler and model assume it.
2. `states.csv` must be sorted by `ts` before windowing/replay (scripts do this).
3. Real fills require the tape; `--tape none` switches to the idealized proxy.
4. Splits are **chronological** (no shuffle) everywhere — never shuffle a time series.
