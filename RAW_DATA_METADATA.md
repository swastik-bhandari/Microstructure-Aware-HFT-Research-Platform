# Raw Dataset Metadata — Orders & Trades

Complete field-level reference for the **raw Hyperliquid streams** consumed by this
project, plus how to verify them on your own machine. Source: Albers et al. (2026),
Zenodo `10.5281/zenodo.18184441` (CC-BY-4.0); field definitions from the dataset's
`SCHEMA.md` and the official `read_data.py` reader.

> **Why this is a separate doc.** `inspect_data.py` reports the *derived* artifacts
> (state table, tape, results). The raw `order_statuses/` and `trades/` folders are
> large and live only on the working machine, so their metadata is documented here and
> verified with `inspect_raw.py`.

---

## How to verify the raw data yourself

Run from the folder containing `order_statuses/`, `trades/`, `mapdir/`:

```bash
python inspect_raw.py                       # one hour (fast)
python inspect_raw.py --date 2025-12-01 --hour 12 --coin eth
```

It prints: physical files on disk, the decoded order DataFrame (rows, columns, status
& TIF breakdowns, price range, unique oids), the decoded trades DataFrame (side
breakdown, px/sz ranges), and the mapdir lookup tables — i.e. the raw data exactly as
the pipeline reads it through `read_data.py`.

---

## 1. Order statuses — `order_statuses/YYYYMMDD/eth_HH.data.gz`

* **Format:** binary, **54 bytes/record**, little-endian packed C struct, records
  concatenated with no delimiter (`filesize / 54 = record count`).
* **Organization:** per-coin (`btc`/`eth`/`sol`), per-hour (`eth_00.data.gz` …
  `eth_23.data.gz`). Accepted orders in `eth_HH.data.gz`; rejected in
  `eth_HH_rejected.data.gz` (not used here).
* **Decoded by:** `read_data.read_orders(order_dir, map_dir, date, hour, coin)`.

### Record fields (byte layout)

| Offset | Field | Type | Size | Units / meaning |
|---:|---|---|---:|---|
| 0 | `ts` | uint64 | 8 | event timestamp, **ns** since Unix epoch |
| 8 | `userId` | uint32 | 4 | pseudonymous trader id → `users.csv` |
| 12 | `isBuilder` | bool | 1 | builder field present |
| 13 | `statusId` | uint8 | 1 | lifecycle status → `statuses.csv` (18 values) |
| 14 | `isAsk` | bool | 1 | **true = Ask/Sell**, false = Bid/Buy |
| 15 | `limitPx` | uint32 | 4 | limit price (**bit-packed** → float) |
| 19 | `sz` | uint32 | 4 | **remaining** size at this event (bit-packed) |
| 23 | `oid` | uint64 | 8 | unique order id (tracks an order across events) |
| 31 | `timestampDiff` | uint32 | 4 | ms since order submission |
| 45 | `tifId` | uint8 | 1 | time-in-force → `tifs.csv` |
| 46 | `triggerPx` | uint32 | 4 | trigger price (bit-packed) |
| 50 | `origSz` | uint32 | 4 | size at submission (bit-packed) |

### Price/size bit-packing (decoded automatically by the reader)
```
decimals = encoded >> 29            # top 3 bits
value    = encoded & 0x1FFFFFFF     # bottom 29 bits
price    = value / 10**decimals     # NOT a simple /1e6
```

### Lifecycle semantics (critical for reconstruction)
* One `oid` appears multiple times: e.g. `open` → one or more `filled` (with
  **decreasing `sz`**) → possibly `canceled` for the remainder.
* Rejected orders appear **exactly once** with a `*Rejected` status and never enter
  the book.
* `origSz` = submission size; `sz` = remaining size at this event. Fresh open:
  `sz == origSz`. Partial fill: `sz < origSz`. Full fill: `sz == 0` *(or the filled
  quantity depending on the event — this ambiguity was resolved empirically: the
  "remaining" interpretation is the one that keeps the book non-crossing).* 

### Status values (`statuses.csv`)
`open`(1), `filled`(5), and cancels: `canceled`(2), `reduceOnlyCanceled`(7),
`scheduledCancel`(10), `siblingFilledCanceled`(11), `selfTradeCanceled`(12),
`marginCanceled`(13), `vaultWithdrawalCanceled`(14), `liquidatedCanceled`(16); plus
`triggered`(9); and 7 rejections: `badAloPxRejected`(0), `perpMarginRejected`(3),
`iocCancelRejected`(4), `minTradeNtlRejected`(6), `reduceOnlyRejected`(8),
`perpMaxPositionRejected`(15), `oracleRejected`(17).

### Time-in-force (`tifs.csv`)
`Alo`(0, post-only), `Gtc`(1), `Ioc`(2), `null`(3), `FrontendMarket`(4),
`LiquidationMarket`(5). **Only `Alo` and `Gtc` limit orders rest in the book.**

### Order types (`order_types.csv`)
`Limit`(0), `Market`(1), `Stop Market`(2), `Take Profit Market`(3),
`Take Profit Limit`(4), `Stop Limit`(5), `Vault Close`(6).

---

## 2. Trades — `trades/YYYYMMDD/HH.gz`

* **Format:** gzip-compressed **JSON lines**, **all coins mixed**, one file per hour
  (`0.gz` … `23.gz`). Filter to `coin == "ETH"`.
* **Decoded by:** `read_data.read_trades(trades_dir, date, hour, coins=["ETH"])`.

### Top-level fields
| Field | Type | Meaning |
|---|---|---|
| `coin` | string | e.g. `"ETH"` — filter key |
| `side` | string | aggressor: **`"A"` = sell-aggressor** (price ↓), **`"B"` = buy-aggressor** (price ↑) |
| `time` | string | ISO ns timestamp → converted to int64 ns |
| `px` | string→float | execution price |
| `sz` | string→float | execution size |
| `hash` | string | L1 tx hash; `"0x000…000"` for same-block trades (not always unique) |
| `trade_dir_override` | string | usually `"Na"` |
| `side_info` | array[2] | the two counterparties (see below) |

### `side_info` (the two counterparties)
| Field | Type | Meaning |
|---|---|---|
| `user` | string | counterparty address |
| `start_pos` | string→float | pre-trade position |
| `oid` | uint64 | counterparty's order id (enables **Level-2 / queue-aware** attribution) |
| `twap_id` | int/null | TWAP id if part of a TWAP order |

> **Used in this project:** `side`, `px`, `sz`, `time` build the trade tape
> (`isB = side=="B"` → the trades that can fill a resting **sell**). `side_info[].oid`
> is *not* used yet — it is the hook for future Level-2 attribution. There is **no
> `tid`** field; dedup uses `hash` (with `(time, oid)` fallback for same-block).

---

## 3. Lookup tables — `mapdir/*.csv`

| File | Rows | Contents |
|---|---|---|
| `order_types.csv` | 7 | order-type id → label |
| `statuses.csv` | 18 | status id → label (incl. 7 rejection types) |
| `tifs.csv` | 6 | time-in-force id → label |
| `users.csv` | ~many (16 MB) | `userId` → Ethereum address (pseudonymous) |
| `value_stats.csv` | ~many (6.8 MB) | per-(key,value) record counts; not used in pipeline |

---

## 4. The one stream NOT used: book diffs

`book_diffs_202512.tar` (≈50 GB) contains every visible-book change and is the stream
*designed* for exact book reconstruction. It was **deliberately skipped** to save
bandwidth/disk; the book is instead reconstructed from the order-status stream (which
uniquely also exposes rejected orders). Consequence: queue position is approximate
(Level-1), documented as a limitation.
