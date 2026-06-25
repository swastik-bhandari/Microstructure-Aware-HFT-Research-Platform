# Order-Size Control as the Active Ingredient in Reinforcement Learning for Microstructure-Aware Trade Execution

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Stable-Baselines3](https://img.shields.io/badge/RL-PPO-green.svg)]()
[![Research](https://img.shields.io/badge/Research-Trade%20Execution-orange.svg)]()

## Overview

This repository contains the implementation and experimental framework for the paper:

**"Order-Size Control as the Active Ingredient in Reinforcement Learning for Microstructure-Aware Trade Execution"**

The project investigates a fundamental question in reinforcement-learning-based optimal trade execution:

> Does reinforcement learning outperform classical execution algorithms because of its policy architecture, or because it can dynamically control order sizes?

Using real Hyperliquid Level-4 ETH order book data, a realistic fill-attribution model, and paired statistical evaluation, we show that **order-size control—not the policy class itself—is the primary source of execution improvement.**

---

## Key Findings

### Main Result

A PPO agent with dynamic order-size selection achieves:

* **39.53 bps** average implementation shortfall
* **2.83 bps improvement over VWAP**
* **60.8% win rate** against VWAP
* **Wilcoxon signed-rank p < 1e-10**
* Consistent improvement across **5 independent training seeds**

### Critical Insight

A PPO agent that controls only:

* order type (market vs limit)
* timing

fails to outperform VWAP.

However, granting the same agent the ability to choose:

* Small orders
* Medium orders
* Large orders

eliminates catastrophic end-of-horizon liquidations and produces statistically significant gains.

---

## Contributions

### 1. Microstructure-Aware Execution Environment

Built using:

* Real Hyperliquid Level-4 order book events
* Reconstructed limit order book states
* Real aggressor trade data
* Honest fill attribution

Unlike simplified simulators, a passive order is credited only when an actual counterparty trade occurred.

### 2. Controlled Action-Space Study

Two PPO agents are compared:

#### PPO Static

Actions:

* Hold
* Limit order
* Market order

Fixed slice size.

#### PPO Size-Aware

Actions:

* Hold
* Limit (small / medium / large)
* Market (small / medium / large)

Dynamic slice sizing.

All other components remain identical:

* Policy architecture
* Features
* Hyperparameters
* Training budget
* Data

This isolates action-space design as the variable under study.

### 3. Rigorous Statistical Evaluation

Experiments use:

* 5 independent seeds
* 300 test windows per seed
* 1,500 paired evaluation windows

Statistical analysis includes:

* Paired t-test
* Wilcoxon signed-rank test
* Sign test
* Bootstrap confidence intervals
* Cohen's d effect size

---

## Dataset

### Market

* Asset: ETH/USDC Perpetual
* Venue: Hyperliquid
* Date: 2025-12-01

### Data Scale

* 86,399 reconstructed one-second snapshots
* 402,483 real trades
* Full Level-4 order book event stream

### Split

| Set   | Percentage |
| ----- | ---------- |
| Train | 70%        |
| Test  | 30%        |

Chronological splitting prevents look-ahead bias.

---

## State Representation

Each environment state contains:

### Market Microstructure Features

1. Bid-ask spread
2. Mid-price
3. Order Book Imbalance (OBI)
4. Market depth
5. Volatility
6. Trade intensity
7. Queue-at-best

### Directional Signal

A frozen CNN-LSTM predicts:

P(up)

using short-horizon order-book dynamics.

### Execution Features

* Remaining inventory fraction
* Remaining time fraction

---

## Market Impact Model

Market-order execution price:

p_exec = mid - spread/2 - impact - fee

where impact follows a nonlinear power-law form:

impact ∝ (size / depth)^α

with:

* 2 bps trading fee
* nonlinear temporary impact

---

## Real Fill Attribution

A limit order fills only when:

* a real buy-aggressor trade occurred
* during the same second
* at or above the posted ask price

This prevents unrealistic passive fills commonly assumed in historical replay simulators.

---

## Reinforcement Learning Setup

### Algorithm

* PPO (Stable-Baselines3)

### Network

* Multi-Layer Perceptron (MLP)

### Hyperparameters

| Parameter      | Value   |
| -------------- | ------- |
| n_steps        | 2048    |
| batch_size     | 256     |
| training_steps | 100,000 |
| seeds          | 5       |

---

## Baselines

The proposed method is compared against:

### VWAP

Volume Weighted Average Price execution using realized volume profiles.

### TWAP

Time Weighted Average Price execution.

### POV

Participation of Volume strategy.

### Passive

Pure limit-order execution.

### PPO Static

Fixed-size reinforcement learning agent.

---

## Results

| Strategy       | Mean Shortfall (bps) |
| -------------- | -------------------- |
| PPO Size-Aware | **39.53**            |
| VWAP           | 42.36                |
| TWAP           | 43.35                |
| POV            | 46.01                |
| PPO Static     | 74.01                |
| Passive        | 169.50               |

Lower is better.

---

## Why PPO Static Fails

The fixed-size agent frequently:

1. Waits too long
2. Misses passive fills
3. Reaches the deadline with inventory remaining
4. Force-liquidates large blocks at market

This creates:

* Severe execution tails
* High variance
* Occasional losses exceeding 300 bps

---

## Why Size-Aware PPO Works

The size-aware agent can:

* Accelerate liquidation when inventory pressure rises
* Use larger slices earlier
* Avoid forced end-dumps
* Reduce tail risk



---

## Main Takeaway

The primary conclusion of this work is:

> In realistic execution environments, action-space expressiveness matters more than policy sophistication.

A fixed-size PPO agent does not beat VWAP.

The same PPO architecture becomes competitive once it can control execution size.

Order-size control is therefore the active ingredient driving reinforcement-learning performance in this execution setting.

---



## License

This repository is released under the MIT License.
