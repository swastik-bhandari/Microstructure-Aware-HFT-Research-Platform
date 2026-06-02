# Hyperliquid Order Flow Dataset

Order lifecycle, order book diff, and trade data from the Hyperliquid decentralized perpetual futures exchange, including the complete message stream with rejected orders. See the accompanying paper for research context:

> **"The 'Neutrinos' of the Order Book: Pervasive, Weakly Interacting Order Flow and its Consequences"**


## Files

### Order Statuses (December 2025)

Complete order lifecycle events for BTC, ETH, and SOL perpetual contracts. Custom binary format, 54 bytes per record. See `SCHEMA.md` for the specification and `read_data.py` for a ready-to-use Python reader.

```
Archive                          Size
─────────────────────────────────────
btc_orders_202512.tar.xz        19 GB
btc_rejected_202512.tar.xz      46 GB
eth_orders_202512.tar.xz        12 GB
eth_rejected_202512.tar.xz      24 GB
sol_orders_202512.tar.xz       6.3 GB
sol_rejected_202512.tar.xz     8.5 GB
```

Coverage: Dec 1–31, 744 hourly files per archive, no gaps. ~880 million records per day.

### Raw Book Diffs (December 2025)

Every change to the visible limit order book for BTC, ETH, and SOL perpetual contracts. Each record identifies the user, order ID, instrument, side, price, and the nature of the change (new order placed, order removed, or size updated after a partial fill). Gzip-compressed JSON lines, one file per hour, all coins interleaved.

```
Archive                          Size
─────────────────────────────────────
book_diffs_202512.tar           50 GB
```

Coverage: Dec 1–31, 24 hourly files per day (744 total), no gaps. The book diff stream records only orders that were accepted onto the book; rejected orders (which never affect the visible book) appear exclusively in the order status data above.

### Trades (October 2025 – January 2026)

Trade-level data for **all coins** on Hyperliquid (250+ perpetual contracts). Gzip-compressed JSON lines, one file per hour.

```
Archive                    Size     Days
─────────────────────────────────────────
trades_2025_10.tar        10 GB      31
trades_2025_11.tar       8.9 GB      30
trades_2025_12.tar       6.7 GB      31
trades_2026_01.tar       3.9 GB      31
```

### Lookup Tables and Documentation

```
File              Description
────────────────────────────────────────────────────────────────────────────
mapdir.tar.xz    CSV lookup tables for decoding binary fields
                  (statuses, order types, TIF, user addresses)
SCHEMA.md        Field-by-field schema documentation
read_data.py     Standalone Python reader (requires NumPy + pandas)
```


## Quick Start

### Extract

```bash
mkdir -p order_statuses book_diffs trades
for f in *_202512.tar.xz; do tar -xf "$f" -C order_statuses/; done
tar -xf book_diffs_202512.tar -C book_diffs/
for f in trades_*.tar; do tar -xf "$f" -C trades/; done
tar -xf mapdir.tar.xz
```

### Read Order Statuses

```python
from read_data import read_orders

# One hour of BTC accepted orders
df = read_orders('order_statuses', 'mapdir', date='2025-12-15', hour=12, coin='btc')

# Rejected orders
df_rej = read_orders('order_statuses', 'mapdir', date='2025-12-15', hour=12,
                     coin='btc', rejected=True)

# Full day, selected columns only (saves memory)
df_day = read_orders('order_statuses', 'mapdir', date='2025-12-15', coin='btc',
                     columns=['ts', 'limitPx', 'sz', 'isAsk', 'statusId'])
```

### Read Book Diffs

```python
from read_data import read_book_diffs

# All coins, one hour
df = read_book_diffs('book_diffs', date='2025-12-15', hour=14)

# BTC only
df_btc = read_book_diffs('book_diffs', date='2025-12-15', hour=14, coins=['BTC'])
```

### Read Trades

```python
from read_data import read_trades

# All coins, one hour
df = read_trades('trades', date='2025-12-15', hour=14)

# BTC only
df_btc = read_trades('trades', date='2025-12-15', hour=14, coins=['BTC'])

# With counterparty info flattened into columns
df = read_trades('trades', date='2025-12-15', hour=14, flatten_side_info=True)
```


## Data Quality

Order status data (Dec 2025) is **complete** — no gaps.

Book diff data (Dec 2025) is **complete** — no gaps.

Trade data has some gaps from collection infrastructure restarts. Data that is present is valid.

```
Month       Complete Days   Missing Hours
─────────────────────────────────────────
Oct 2025        29/31              2
Nov 2025        25/30             10
Dec 2025        31/31              0
Jan 2026        30/31              1
```


## Citation

```bibtex
@dataset{hyperliquid_order_flow_2026,
  title     = {An Open Book: Level 4 Order Book Data from the Hyperliquid Exchange},
  author    = {Albers, Jakob and Cucuringu, Mihai and Howison, Sam and Shestopaloff, Alexander Y.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18184441}
}
```

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
