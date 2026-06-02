# Schema Documentation

This document provides the complete field-level specification for all three data types in this dataset: **order statuses** (binary format), **trades** (JSON lines), and **raw book diffs** (JSON lines).


## 1. Order Status Data

### File Organization

Order status data is organized in a two-level hierarchy:

```
<date>/                          # Daily folder, e.g. 20251215
    <coin>_<HH>.data.gz          # Accepted orders for hour HH
    <coin>_<HH>_rejected.data.gz # Rejected orders for hour HH
```

- **Coins:** `btc`, `eth`, `sol`
- **Hours:** `00` through `23` (UTC)
- Each file is gzip-compressed binary data

### Binary Record Format

Each record is exactly **54 bytes**, stored as a packed C struct with little-endian byte order. Records are concatenated without delimiters; the file size divided by 54 gives the exact record count.

| Offset | Field | Type | Size | Description |
|--------|-------|------|------|-------------|
| 0 | `ts` | uint64 | 8 | Event timestamp (nanoseconds since Unix epoch) |
| 8 | `userId` | uint32 | 4 | User identifier (see `users.csv`) |
| 12 | `isBuilder` | bool | 1 | Whether the builder field was present |
| 13 | `statusId` | uint8 | 1 | Order status (see `statuses.csv`) |
| 14 | `isAsk` | bool | 1 | `true` = Ask/Sell, `false` = Bid/Buy |
| 15 | `limitPx` | uint32 | 4 | Limit price (encoded; see Price Encoding) |
| 19 | `sz` | uint32 | 4 | Current order size (encoded; see Price Encoding) |
| 23 | `oid` | uint64 | 8 | Unique order identifier |
| 31 | `timestampDiff` | uint32 | 4 | `ts` minus original order submission timestamp, in milliseconds |
| 35 | `triggerCondition` | int32 | 4 | Trigger price condition (signed encoding; see below) |
| 39 | `triggered` | bool | 1 | Whether the trigger condition was met |
| 40 | `isTrigger` | bool | 1 | Whether this is a trigger/conditional order |
| 41 | `hasChildren` | bool | 1 | Whether the order has child TP/SL orders |
| 42 | `isPositionTpsl` | bool | 1 | Whether this is a position-level TP/SL |
| 43 | `reduceOnly` | bool | 1 | Whether the order is reduce-only |
| 44 | `orderTypeId` | uint8 | 1 | Order type (see `order_types.csv`) |
| 45 | `tifId` | uint8 | 1 | Time-in-force (see `tifs.csv`) |
| 46 | `triggerPx` | uint32 | 4 | Trigger price (encoded; see Price Encoding) |
| 50 | `origSz` | uint32 | 4 | Original order size at submission (encoded) |

The corresponding NumPy dtype for reading:

```python
import numpy as np

RECORD_DTYPE = np.dtype([
    ('ts',              '<u8'),
    ('userId',          '<u4'),
    ('isBuilder',        '?'),
    ('statusId',        '<u1'),
    ('isAsk',            '?'),
    ('limitPx',         '<u4'),
    ('sz',              '<u4'),
    ('oid',             '<u8'),
    ('timestampDiff',   '<u4'),
    ('triggerCondition','<i4'),
    ('triggered',        '?'),
    ('isTrigger',        '?'),
    ('hasChildren',      '?'),
    ('isPositionTpsl',   '?'),
    ('reduceOnly',       '?'),
    ('orderTypeId',     '<u1'),
    ('tifId',           '<u1'),
    ('triggerPx',       '<u4'),
    ('origSz',          '<u4'),
])

assert RECORD_DTYPE.itemsize == 54
```

### Price Encoding

Prices and sizes (`limitPx`, `sz`, `origSz`, `triggerPx`) use a custom fixed-point encoding that packs a floating-point value into a uint32:

```
Bits 31–29 (3 bits): number of decimal places (0–7)
Bits 28–0  (29 bits): integer value
```

**Decoding formula:**

```
decoded = value / 10^decimals
```

where `value = encoded & 0x1FFFFFFF` and `decimals = encoded >> 29`.

**Example:** For BTC priced at $96,543.21:
- `decimals = 2`, `value = 9654321`
- `encoded = (2 << 29) | 9654321 = 0x40935031`

**Python implementation:**

```python
def decode_price(encoded):
    """Decode uint32-encoded prices to float32."""
    powers = np.array([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=np.float32)
    decimals = encoded >> 29
    value = encoded & 0x1FFFFFFF
    return (value / powers[decimals]).astype(np.float32)
```

### Signed Price Encoding

The `triggerCondition` field uses a signed variant:

```
Bits 31–29 (3 bits): number of decimal places
Bit  28    (1 bit):  sign (1 = negative / "price below", 0 = positive / "price above")
Bits 27–0  (28 bits): integer value
```

**Python implementation:**

```python
def decode_signed_price(encoded):
    """Decode signed price encoding (for triggerCondition).
    Input must be uint32 (cast first if stored as int32)."""
    encoded = np.asarray(encoded, dtype=np.uint32)
    powers = np.array([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=np.float32)
    decimals = encoded >> 29
    value = encoded & 0x0FFFFFFF
    is_negative = (encoded & 0x10000000) != 0
    price = value.astype(np.float32) / powers[decimals]
    return np.where(is_negative, -price, price)
```

### Timestamp Conventions

- **`ts`**: Nanoseconds since Unix epoch (1970-01-01T00:00:00Z). Divide by 1e9 for seconds, or cast directly: `np.array(ts, dtype='datetime64[ns]')`.
- **`timestampDiff`**: The difference `ts - order_creation_time` in **milliseconds**. For newly opened orders, this is typically 0. For cancellations and fills, it indicates how long the order lived.

### Field Notes

**`statusId`** — Each record represents a single event in an order's lifecycle. An order may appear multiple times with different statuses:
- A typical lifecycle: `open` (accepted into book) then `filled` or `canceled`
- An order that is partially filled will appear as `open`, then one or more `filled` events (with decreasing `sz`), then possibly `canceled` for the remainder
- Rejected orders appear exactly once with status `badAloPxRejected`

**`oid`** — Unique order identifier assigned by the matching engine. Use this to track an order across its lifecycle events. Multiple records with the same `oid` represent successive events for the same order.

**`userId`** — Integer identifier mapping to an Ethereum address via `users.csv`. Pseudonymous: the same address may represent an individual, a bot, or a sub-account of a larger operation.

**`isAsk`** — Indicates the order side. `true` = the order is selling (ask/offer side). `false` = the order is buying (bid side).

**`origSz` vs `sz`** — `origSz` is the size at original submission. `sz` is the remaining size at the time of this event. For a fresh open, `sz == origSz`. For a partial fill, `sz < origSz`. For a full fill, `sz == 0` (or the filled quantity, depending on the event).


## 2. Lookup Tables (mapdir)

### statuses.csv

Maps `statusId` (uint8) to human-readable status labels.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `badAloPxRejected` | Post-only order rejected: would have crossed the spread |
| 1 | `open` | Order accepted into the order book |
| 2 | `canceled` | Order canceled by the trader |
| 3 | `perpMarginRejected` | Rejected: insufficient margin |
| 4 | `iocCancelRejected` | IOC order: unfilled portion canceled |
| 5 | `filled` | Order filled (fully or partially) |
| 6 | `minTradeNtlRejected` | Rejected: below minimum trade notional |
| 7 | `reduceOnlyCanceled` | Reduce-only order canceled (no position to reduce) |
| 8 | `reduceOnlyRejected` | Reduce-only order rejected |
| 9 | `triggered` | Trigger/stop order activated |
| 10 | `scheduledCancel` | Order canceled by scheduled/time-based cancellation |
| 11 | `siblingFilledCanceled` | Sibling order (e.g., other leg of OCO) filled; this one canceled |
| 12 | `selfTradeCanceled` | Canceled to prevent self-trade |
| 13 | `marginCanceled` | Canceled due to margin constraints |
| 14 | `vaultWithdrawalCanceled` | Canceled due to vault withdrawal |
| 15 | `perpMaxPositionRejected` | Rejected: would exceed maximum position size |
| 16 | `liquidatedCanceled` | Canceled as part of a liquidation |
| 17 | `oracleRejected` | Rejected: oracle price constraint violated |

### order_types.csv

Maps `orderTypeId` (uint8) to order type labels.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `Limit` | Standard limit order |
| 1 | `Market` | Market order |
| 2 | `Stop Market` | Stop-loss market order |
| 3 | `Take Profit Market` | Take-profit market order |
| 4 | `Take Profit Limit` | Take-profit limit order |
| 5 | `Stop Limit` | Stop-loss limit order |
| 6 | `Vault Close` | Vault close order |

### tifs.csv

Maps `tifId` (uint8) to time-in-force labels.

| ID | Label | Description |
|----|-------|-------------|
| 0 | `Alo` | Add Liquidity Only (post-only). Rejected if it would cross the spread. |
| 1 | `Gtc` | Good Till Canceled. Rests in the book until filled or canceled. |
| 2 | `Ioc` | Immediate Or Cancel. Fills what it can, cancels the rest. |
| 3 | `null` | Not specified (typically for trigger/conditional orders before activation). |
| 4 | `FrontendMarket` | Market order submitted via the Hyperliquid frontend UI. |
| 5 | `LiquidationMarket` | Market order generated by the liquidation engine. |

### users.csv

Maps `userId` (uint32) to Ethereum addresses (hex strings). Format: `<address>,<id>`.

This file contains 328,456 entries. User IDs are assigned incrementally as new addresses interact with the exchange.


## 3. Trade Data

### File Organization

```
<date>/          # Daily folder, e.g. 20250701
    <H>.gz       # Trades for hour H (0–23), gzip-compressed JSON lines
```

Hours are numbered 0–23 (not zero-padded in filenames).

### JSON Schema

Each line in a trade file is a JSON object representing a single trade:

```json
{
  "coin": "BTC",
  "side": "A",
  "time": "2025-07-01T12:34:56.789012345",
  "px": "96543.0",
  "sz": "0.125",
  "hash": "0x1234...abcd",
  "trade_dir_override": "Na",
  "side_info": [
    {
      "user": "0xabc...123",
      "start_pos": "1.5",
      "oid": 110237687900,
      "twap_id": null,
      "cloid": null
    },
    {
      "user": "0xdef...456",
      "start_pos": "-2.0",
      "oid": 110237699140,
      "twap_id": 857226,
      "cloid": null
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `coin` | string | Trading pair symbol (e.g., `"BTC"`, `"ETH"`, `"SOL"`, `"HYPE"`) |
| `side` | string | Aggressor side. `"A"` = sell aggressor (price moved down), `"B"` = buy aggressor (price moved up) |
| `time` | string | Trade timestamp in ISO 8601 format with nanosecond precision |
| `px` | string | Execution price (as a decimal string to preserve precision) |
| `sz` | string | Execution size in base asset units (as a decimal string) |
| `hash` | string | L1 transaction hash. `"0x000...000"` for trades within the same block as the order submission |
| `trade_dir_override` | string | Direction override indicator. Usually `"Na"` |
| `side_info` | array[2] | The two counterparties to the trade (see below) |

### side_info Fields

Each trade has exactly two entries in `side_info`: the two counterparties.

| Field | Type | Description |
|-------|------|-------------|
| `user` | string | Ethereum address of the counterparty |
| `start_pos` | string | Position size of this user *before* the trade (signed; negative = short) |
| `oid` | integer | Order ID that generated this fill |
| `twap_id` | integer or null | TWAP order ID if part of a TWAP execution, else `null` |
| `cloid` | string or null | Client order ID (hex string) if provided by the trader, else `null` |

### Notes on Trade Data

- **All coins included:** Trade files contain every perpetual contract listed on Hyperliquid (250+ as of late 2025), not just BTC/ETH/SOL. Filter on the `coin` field to select specific instruments.
- **Numeric strings:** `px` and `sz` are encoded as strings rather than numbers to preserve decimal precision. Cast to float for analysis: `df['px'] = df['px'].astype(float)`.
- **Counterparty information:** The `side_info` array provides full counterparty detail for each trade, including pre-trade positions and order IDs. This enables analysis of informed vs. uninformed flow, TWAP participation, and position-level dynamics.
- **Timestamp precision:** Nanosecond precision, reflecting the L1 block timestamp.


## 4. Raw Book Diff Data

### File Organization

Book diff data is organized in a two-level hierarchy inside the archive:

```
<date>/          # Daily folder, e.g. 20251215
    ex<H>.gz     # Book diffs for hour H (0–23), gzip-compressed JSON lines
```

- **Coins:** BTC, ETH, and SOL are interleaved in each file (filter on the `coin` field)
- **Hours:** Numbered 0–23 (not zero-padded in filenames)
- Each file is gzip-compressed JSON lines (one record per line)

### Relationship to Order Status Data

The book diff stream records every change to the **visible** limit order book. It captures only orders that were accepted onto the book and their subsequent removal or modification. Rejected orders — which constitute approximately 89% of all order submissions and are the focus of the companion paper — never appear in the book diff stream because they are rejected by the matching engine before affecting the book. Rejected orders are recorded exclusively in the order status data (Section 1).

Together, the two streams provide a complete picture: the order status data captures every order *attempt* (including rejections), while the book diff data captures every resulting change to the order book state.

### JSON Schema

Each line in a book diff file is a JSON object representing a single book change. There are three types:

**New order placed on the book:**

```json
{
  "user": "0x31ca8395cf837de08b24da3f660e77761dfb974b",
  "oid": 273024455731,
  "coin": "ETH",
  "side": "A",
  "px": "2964.0",
  "raw_book_diff": {"new": {"sz": "0.2274"}}
}
```

**Order removed from the book** (cancellation or full fill):

```json
{
  "user": "0x6ba889db7f923622d3548f621ecc2054b80c1817",
  "oid": 273024450830,
  "coin": "ETH",
  "side": "B",
  "px": "2962.3",
  "raw_book_diff": "remove"
}
```

**Order size updated** (partial fill):

```json
{
  "user": "0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00",
  "oid": 273024455174,
  "coin": "ETH",
  "side": "B",
  "px": "2962.5",
  "raw_book_diff": {"update": {"origSz": "5.0632", "newSz": "4.7287"}}
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `user` | string | Ethereum address of the order owner |
| `oid` | integer | Unique order identifier (links to `oid` in the order status data) |
| `coin` | string | Instrument symbol: `"BTC"`, `"ETH"`, or `"SOL"` |
| `side` | string | Order side: `"A"` = ask/sell, `"B"` = bid/buy |
| `px` | string | Price level of the order (decimal string) |
| `raw_book_diff` | string or object | The nature of the change (see below) |

### raw_book_diff Values

| Value | Type | Description |
|-------|------|-------------|
| `"remove"` | string | Order was removed from the book (cancelled or fully filled). Approximately 50% of records. |
| `{"new": {"sz": "..."}}` | object | New order placed on the book at this price level, with the given size. Approximately 50% of records. |
| `{"update": {"origSz": "...", "newSz": "..."}}` | object | Order size changed due to a partial fill. `origSz` is the size before the fill, `newSz` is the remaining size. Less than 1% of records. |

### Notes on Book Diff Data

- **Coin distribution:** BTC accounts for roughly 50% of records, ETH for roughly 32%, and SOL for roughly 18%.
- **Record volume:** A typical hour contains 3–10 million records, depending on market activity.
- **Numeric strings:** `px`, `sz`, `origSz`, and `newSz` are encoded as strings to preserve decimal precision. Cast to float for analysis.
- **Reconstructing the order book:** By processing book diff records in sequence, one can reconstruct the complete state of the limit order book at any point in time. Start from an empty book and apply each `new`, `remove`, and `update` event in order. The `oid` field uniquely identifies each resting order.
- **Linking to order statuses:** The `oid` field in the book diff data corresponds to the `oid` field in the order status data, allowing researchers to join the two streams. For example, one can determine the exact order type, time-in-force, and submission timestamp of any order that appears on the book.
