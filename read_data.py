"""
Hyperliquid Order Flow Dataset — Reader
========================================

Standalone reader for the Hyperliquid order flow dataset.
Requires only NumPy and pandas (Python >= 3.9).

Usage:
    from read_data import read_orders, read_trades, read_book_diffs

    # Load one hour of BTC accepted orders
    df = read_orders('order_statuses', 'mapdir', date='2025-12-15', hour=12, coin='btc')

    # Load one hour of BTC rejected orders
    df = read_orders('order_statuses', 'mapdir', date='2025-12-15', hour=12,
                     coin='btc', rejected=True)

    # Load a full day (all 24 hours)
    df = read_orders('order_statuses', 'mapdir', date='2025-12-15', coin='btc')

    # Load only specific columns (saves memory)
    df = read_orders('order_statuses', 'mapdir', date='2025-12-15', hour=12,
                     coin='btc', columns=['ts', 'limitPx', 'sz', 'isAsk'])

    # Load book diffs for one hour (all coins)
    df = read_book_diffs('book_diffs', date='2025-12-15', hour=14)

    # Load book diffs filtered to specific coins
    df = read_book_diffs('book_diffs', date='2025-12-15', hour=14, coins=['BTC', 'ETH'])

    # Load trades for one hour (all coins)
    df = read_trades('trades', date='2025-12-15', hour=14)

    # Load trades filtered to specific coins
    df = read_trades('trades', date='2025-12-15', hour=14, coins=['BTC', 'ETH'])

    # List available dates
    dates = list_available_dates('order_statuses', coin='btc')
"""

import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "read_orders",
    "read_book_diffs",
    "read_trades",
    "list_available_dates",
    "list_available_book_diff_dates",
    "list_available_trade_dates",
    "RECORD_DTYPE",
]


# ---------------------------------------------------------------------------
# Binary record format (54 bytes per record)
# ---------------------------------------------------------------------------

RECORD_DTYPE = np.dtype([
    ("ts",              "<u8"),   # Timestamp: nanoseconds since Unix epoch
    ("userId",          "<u4"),   # User ID (maps to address via users.csv)
    ("isBuilder",        "?"),    # Builder field present in original data
    ("statusId",        "<u1"),   # Order status (maps via statuses.csv)
    ("isAsk",            "?"),    # True = Ask/Sell, False = Bid/Buy
    ("limitPx",         "<u4"),   # Limit price (encoded)
    ("sz",              "<u4"),   # Current size (encoded)
    ("oid",             "<u8"),   # Order ID
    ("timestampDiff",   "<u4"),   # Event time minus order creation time (ms)
    ("triggerCondition", "<i4"),  # Trigger condition (signed encoding)
    ("triggered",        "?"),    # Trigger condition was met
    ("isTrigger",        "?"),    # Is a trigger/conditional order
    ("hasChildren",      "?"),    # Has child TP/SL orders
    ("isPositionTpsl",   "?"),    # Position-level TP/SL
    ("reduceOnly",       "?"),    # Reduce-only order
    ("orderTypeId",     "<u1"),   # Order type (maps via order_types.csv)
    ("tifId",           "<u1"),   # Time-in-force (maps via tifs.csv)
    ("triggerPx",       "<u4"),   # Trigger price (encoded)
    ("origSz",          "<u4"),   # Original size at submission (encoded)
])

assert RECORD_DTYPE.itemsize == 54


# ---------------------------------------------------------------------------
# Price decoding
# ---------------------------------------------------------------------------

_POWERS = np.array([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=np.float32)


def decode_price(encoded: np.ndarray) -> np.ndarray:
    """Decode uint32-encoded prices/sizes to float32.

    Encoding: top 3 bits = number of decimal places (0-7),
              bottom 29 bits = integer value.
    Decoded = value / 10^decimals.
    """
    decimals = encoded >> 29
    value = encoded & 0x1FFFFFFF
    return (value / _POWERS[decimals]).astype(np.float32)


def decode_signed_price(encoded: np.ndarray) -> np.ndarray:
    """Decode signed price encoding (used for triggerCondition).

    Encoding: bits 31-29 = decimals, bit 28 = sign, bits 27-0 = value.
    Positive = "price above X", negative = "price below X".
    """
    decimals = encoded >> 29
    value = encoded & 0x0FFFFFFF
    is_negative = (encoded & 0x10000000) != 0
    price = value.astype(np.float32) / _POWERS[decimals]
    return np.where(is_negative, -price, price)


# ---------------------------------------------------------------------------
# Lookup table loading
# ---------------------------------------------------------------------------

def load_category_maps(mapdir: Union[str, Path]) -> Dict[str, Dict[int, str]]:
    """Load all category mapping CSVs from the mapdir directory.

    Returns a dict with keys 'statuses', 'order_types', 'tifs', each mapping
    integer IDs to human-readable labels.
    """
    mapdir = Path(mapdir)
    maps = {}
    for name in ("statuses", "order_types", "tifs"):
        path = mapdir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, header=None, names=["label", "id"],
                             keep_default_na=False)
            maps[name] = dict(zip(df["id"], df["label"]))
    return maps


def load_user_map(mapdir: Union[str, Path]) -> Dict[int, str]:
    """Load user ID to Ethereum address mapping.

    Returns a dict mapping integer user IDs to hex address strings.
    Only load this if you need address resolution — it contains 328K entries.
    """
    path = Path(mapdir) / "users.csv"
    df = pd.read_csv(path, header=None, names=["address", "id"],
                     keep_default_na=False)
    return dict(zip(df["id"], df["address"]))


# ---------------------------------------------------------------------------
# Order status reading
# ---------------------------------------------------------------------------

def _read_single_hour(
    data_dir: Path,
    date_str: str,
    hour: int,
    coin: str,
    rejected: bool,
    columns: Optional[Sequence[str]],
) -> Optional[pd.DataFrame]:
    """Read a single hourly binary file into a DataFrame."""
    suffix = "_rejected" if rejected else ""
    filepath = data_dir / date_str / f"{coin}_{hour:02d}{suffix}.data.gz"
    if not filepath.exists():
        return None

    with gzip.open(filepath, "rb") as f:
        raw = f.read()

    n_records = len(raw) // RECORD_DTYPE.itemsize
    if n_records == 0:
        return None

    data = np.frombuffer(raw, dtype=RECORD_DTYPE, count=n_records)

    # Select columns
    if columns is not None:
        cols = [c for c in columns if c in RECORD_DTYPE.names]
        if "ts" not in cols:
            cols = ["ts"] + cols
    else:
        cols = list(RECORD_DTYPE.names)

    df = pd.DataFrame({col: data[col] for col in cols})

    # Decode timestamps
    df["ts"] = df["ts"].astype("datetime64[ns]")

    # Decode encoded price/size fields
    for col in ("limitPx", "triggerPx", "sz", "origSz"):
        if col in df.columns:
            df[col] = decode_price(df[col].to_numpy(dtype=np.uint32))

    if "triggerCondition" in df.columns:
        df["triggerCondition"] = decode_signed_price(
            df["triggerCondition"].to_numpy(dtype=np.uint32)
        )

    return df


def read_orders(
    data_dir: Union[str, Path],
    mapdir: Union[str, Path],
    date: str,
    hour: Optional[int] = None,
    coin: str = "btc",
    rejected: bool = False,
    columns: Optional[Sequence[str]] = None,
    apply_labels: bool = True,
) -> pd.DataFrame:
    """Read order status data for a given date, coin, and type.

    Args:
        data_dir:     Path to the extracted order status directory.
        mapdir:       Path to the extracted mapdir directory.
        date:         Date string in 'YYYY-MM-DD' or 'YYYYMMDD' format.
        hour:         Hour to load (0-23). If None, loads all 24 hours.
        coin:         Coin to load: 'btc', 'eth', or 'sol'.
        rejected:     If True, load rejected orders; if False, accepted orders.
        columns:      List of column names to load (None = all). Reduces memory.
        apply_labels: If True, add human-readable 'status', 'orderType', 'tif'
                      columns by mapping the integer ID fields.

    Returns:
        DataFrame with decoded order data. Raises ValueError if no data found.

    Example:
        >>> df = read_orders('order_statuses', 'mapdir',
        ...                  date='2025-12-15', hour=12, coin='btc')
        >>> print(f"{len(df):,} records")
        5,195,978 records
    """
    data_dir = Path(data_dir)
    mapdir = Path(mapdir)
    coin = coin.lower()

    # Normalize date format
    date_str = date.replace("-", "")
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date format: '{date}'. Use 'YYYY-MM-DD' or 'YYYYMMDD'.")

    # Determine hours to load
    hours = [hour] if hour is not None else list(range(24))

    # Read and concatenate
    frames = []
    for h in hours:
        df = _read_single_hour(data_dir, date_str, h, coin, rejected, columns)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError(
            f"No data found for {coin} {'rejected' if rejected else 'orders'} "
            f"on {date}" + (f" hour {hour}" if hour is not None else "")
        )

    result = pd.concat(frames, ignore_index=True)

    # Apply human-readable labels
    if apply_labels:
        maps = load_category_maps(mapdir)
        label_mapping = {
            "statusId":    ("statuses",    "status"),
            "orderTypeId": ("order_types", "orderType"),
            "tifId":       ("tifs",        "tif"),
        }
        for id_col, (map_name, new_col) in label_mapping.items():
            if id_col in result.columns and map_name in maps:
                result[new_col] = result[id_col].map(maps[map_name])

    return result


# ---------------------------------------------------------------------------
# Trade reading
# ---------------------------------------------------------------------------

def read_trades(
    data_dir: Union[str, Path],
    date: str,
    hour: Optional[int] = None,
    coins: Optional[Sequence[str]] = None,
    flatten_side_info: bool = False,
) -> pd.DataFrame:
    """Read trade data for a given date.

    Args:
        data_dir:          Path to the extracted trades directory.
        date:              Date string in 'YYYY-MM-DD' or 'YYYYMMDD' format.
        hour:              Hour to load (0-23). If None, loads all 24 hours.
        coins:             List of coin symbols to include (e.g., ['BTC', 'ETH']).
                           Case-insensitive. If None, loads all coins.
        flatten_side_info: If True, flatten the side_info array into separate
                           columns (side0_user, side1_user, etc.). If False,
                           keep side_info as a list column.

    Returns:
        DataFrame with trade data. Raises ValueError if no data found.

    Example:
        >>> df = read_trades('trades', date='2025-12-15', hour=14, coins=['BTC'])
        >>> print(f"{len(df):,} BTC trades")
    """
    data_dir = Path(data_dir)
    date_str = date.replace("-", "")
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date format: '{date}'. Use 'YYYY-MM-DD' or 'YYYYMMDD'.")

    hours = [hour] if hour is not None else list(range(24))
    coin_filter = {c.upper() for c in coins} if coins else None

    frames = []
    for h in hours:
        filepath = data_dir / date_str / f"{h}.gz"
        if not filepath.exists():
            continue

        with gzip.open(filepath, "rb") as f:
            raw = f.read()

        lines = raw.decode("utf-8").strip().split("\n")
        if not lines or lines == [""]:
            continue

        records = []
        for line in lines:
            trade = json.loads(line)
            if coin_filter and trade["coin"].upper() not in coin_filter:
                continue
            records.append(trade)

        if not records:
            continue

        df = pd.DataFrame(records)
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No trade data found for {date}"
            + (f" hour {hour}" if hour is not None else "")
            + (f" coins {list(coins)}" if coins else "")
        )

    result = pd.concat(frames, ignore_index=True)

    # Convert numeric string fields to proper types
    result["px"] = pd.to_numeric(result["px"], errors="coerce")
    result["sz"] = pd.to_numeric(result["sz"], errors="coerce")
    result["time"] = pd.to_datetime(result["time"])

    # Flatten side_info if requested
    if flatten_side_info and "side_info" in result.columns:
        result = _flatten_side_info(result)

    return result


def _flatten_side_info(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the side_info array into separate columns for each counterparty."""
    side_0 = pd.json_normalize(df["side_info"].apply(lambda x: x[0] if len(x) > 0 else {}))
    side_1 = pd.json_normalize(df["side_info"].apply(lambda x: x[1] if len(x) > 1 else {}))

    side_0.columns = [f"side0_{c}" for c in side_0.columns]
    side_1.columns = [f"side1_{c}" for c in side_1.columns]

    result = pd.concat([df.drop(columns=["side_info"]), side_0, side_1], axis=1)
    return result


# ---------------------------------------------------------------------------
# Book diff reading
# ---------------------------------------------------------------------------

def read_book_diffs(
    data_dir: Union[str, Path],
    date: str,
    hour: Optional[int] = None,
    coins: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read raw book diff data for a given date.

    Each record represents a single change to the visible limit order book:
    a new order placed, an order removed, or an order size updated (partial fill).

    Args:
        data_dir: Path to the extracted book_diffs directory.
        date:     Date string in 'YYYY-MM-DD' or 'YYYYMMDD' format.
        hour:     Hour to load (0-23). If None, loads all 24 hours.
        coins:    List of coin symbols to include (e.g., ['BTC', 'ETH']).
                  Case-insensitive. If None, loads all coins.

    Returns:
        DataFrame with columns: user, oid, coin, side, px, diff_type,
        sz (for 'new'), orig_sz and new_sz (for 'update').
        Raises ValueError if no data found.

    Example:
        >>> df = read_book_diffs('book_diffs', date='2025-12-15', hour=14)
        >>> print(df['diff_type'].value_counts())
    """
    data_dir = Path(data_dir)
    date_str = date.replace("-", "")
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date format: '{date}'. Use 'YYYY-MM-DD' or 'YYYYMMDD'.")

    hours = [hour] if hour is not None else list(range(24))
    coin_filter = {c.upper() for c in coins} if coins else None

    frames = []
    for h in hours:
        filepath = data_dir / date_str / f"ex{h}.gz"
        if not filepath.exists():
            continue

        with gzip.open(filepath, "rb") as f:
            raw = f.read()

        lines = raw.decode("utf-8").strip().split("\n")
        if not lines or lines == [""]:
            continue

        records = []
        for line in lines:
            rec = json.loads(line)
            if coin_filter and rec["coin"].upper() not in coin_filter:
                continue

            rbd = rec["raw_book_diff"]
            row = {
                "user": rec["user"],
                "oid": rec["oid"],
                "coin": rec["coin"],
                "side": rec["side"],
                "px": rec["px"],
            }

            if rbd == "remove":
                row["diff_type"] = "remove"
            elif isinstance(rbd, dict):
                if "new" in rbd:
                    row["diff_type"] = "new"
                    row["sz"] = rbd["new"]["sz"]
                elif "update" in rbd:
                    row["diff_type"] = "update"
                    row["orig_sz"] = rbd["update"]["origSz"]
                    row["new_sz"] = rbd["update"]["newSz"]

            records.append(row)

        if not records:
            continue

        df = pd.DataFrame(records)
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No book diff data found for {date}"
            + (f" hour {hour}" if hour is not None else "")
            + (f" coins {list(coins)}" if coins else "")
        )

    result = pd.concat(frames, ignore_index=True)

    # Convert numeric string fields to proper types
    result["px"] = pd.to_numeric(result["px"], errors="coerce")
    for col in ("sz", "orig_sz", "new_sz"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def list_available_dates(
    data_dir: Union[str, Path],
    coin: str = "btc",
    rejected: bool = False,
) -> List[str]:
    """List all dates that have order status data for a given coin.

    Returns sorted list of date strings in 'YYYYMMDD' format.
    """
    data_dir = Path(data_dir)
    suffix = "_rejected" if rejected else ""
    dates = set()
    for date_dir in data_dir.iterdir():
        if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
            # Check if at least one hourly file exists
            sample = date_dir / f"{coin.lower()}_00{suffix}.data.gz"
            if sample.exists():
                dates.add(date_dir.name)
    return sorted(dates)


def list_available_book_diff_dates(data_dir: Union[str, Path]) -> List[str]:
    """List all dates that have book diff data.

    Returns sorted list of date strings in 'YYYYMMDD' format.
    """
    data_dir = Path(data_dir)
    dates = set()
    for date_dir in data_dir.iterdir():
        if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
            if any(date_dir.glob("ex*.gz")):
                dates.add(date_dir.name)
    return sorted(dates)


def list_available_trade_dates(data_dir: Union[str, Path]) -> List[str]:
    """List all dates that have trade data.

    Returns sorted list of date strings in 'YYYYMMDD' format.
    """
    data_dir = Path(data_dir)
    dates = set()
    for date_dir in data_dir.iterdir():
        if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
            if any(date_dir.glob("*.gz")):
                dates.add(date_dir.name)
    return sorted(dates)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print()
    print("=" * 65)
    print("  Hyperliquid Order Flow Dataset — Reader Demo")
    print("=" * 65)

    # Auto-detect directories
    script_dir = Path(__file__).parent
    order_dir = None
    book_diff_dir = None
    trade_dir = None
    map_dir = None

    # Try common locations
    for candidate in [script_dir, script_dir / "order_statuses", Path("order_statuses")]:
        if candidate.is_dir() and any(candidate.glob("202*")):
            order_dir = candidate
            break

    for candidate in [script_dir / "book_diffs", Path("book_diffs"),
                      script_dir.parent / "book_diffs"]:
        if candidate.is_dir() and any(candidate.glob("202*")):
            book_diff_dir = candidate
            break

    for candidate in [script_dir, script_dir / "trades", Path("trades")]:
        if candidate.is_dir() and any(candidate.glob("202*")):
            trade_dir = candidate
            break

    for candidate in [script_dir / "mapdir", Path("mapdir"),
                      script_dir.parent / "mapdir"]:
        if candidate.is_dir() and (candidate / "statuses.csv").exists():
            map_dir = candidate
            break

    # --- Order status demo ---
    if order_dir and map_dir:
        dates = list_available_dates(order_dir, coin="btc")
        print(f"\n  Order status data: {len(dates)} dates available")
        print(f"  Range: {dates[0]} to {dates[-1]}")
        print(f"  Data dir: {order_dir}")
        print(f"  Map dir:  {map_dir}")

        # Load a sample hour
        sample_date = dates[len(dates) // 2]  # middle of the range
        print(f"\n  Loading BTC accepted orders for {sample_date} hour 12...")
        df = read_orders(order_dir, map_dir, date=sample_date, hour=12, coin="btc")
        print(f"  Loaded {len(df):,} records ({df.memory_usage(deep=True).sum()/1e6:.0f} MB)")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"\n  Status distribution:")
        for status, count in df["status"].value_counts().head(5).items():
            print(f"    {status:30s}  {count:>10,}")

        print(f"\n  Loading BTC rejected orders for {sample_date} hour 12...")
        df_r = read_orders(order_dir, map_dir, date=sample_date, hour=12,
                           coin="btc", rejected=True)
        print(f"  Loaded {len(df_r):,} records")
        print(f"  Rejected/accepted ratio: {len(df_r)/len(df):.1f}x")

        print(f"\n  Price range: ${df['limitPx'].min():,.2f} – ${df['limitPx'].max():,.2f}")
        print(f"  Median size: {df['sz'].median():.4f}")
    else:
        print("\n  Order status data not found.")
        print("  Expected: directory with 'YYYYMMDD' subfolders containing .data.gz files")
        if not map_dir:
            print("  Mapdir not found — expected: directory with statuses.csv, etc.")

    # --- Book diff demo ---
    if book_diff_dir:
        dates = list_available_book_diff_dates(book_diff_dir)
        print(f"\n  {'—' * 50}")
        print(f"  Book diff data: {len(dates)} dates available")
        print(f"  Range: {dates[0]} to {dates[-1]}")
        print(f"  Data dir: {book_diff_dir}")

        sample_date = dates[len(dates) // 2]
        print(f"\n  Loading all book diffs for {sample_date} hour 14...")
        try:
            df_bd = read_book_diffs(book_diff_dir, date=sample_date, hour=14)
            print(f"  Loaded {len(df_bd):,} records")
            print(f"\n  Diff type distribution:")
            for dtype, count in df_bd["diff_type"].value_counts().items():
                print(f"    {dtype:15s}  {count:>10,}  ({100*count/len(df_bd):.1f}%)")
            print(f"\n  Coin distribution:")
            for coin, count in df_bd["coin"].value_counts().items():
                print(f"    {coin:15s}  {count:>10,}  ({100*count/len(df_bd):.1f}%)")
        except ValueError as e:
            print(f"  {e}")
    else:
        print("\n  Book diff data not found.")
        print("  Expected: directory with 'YYYYMMDD' subfolders containing ex*.gz files")

    # --- Trade demo ---
    if trade_dir:
        dates = list_available_trade_dates(trade_dir)
        print(f"\n  {'—' * 50}")
        print(f"  Trade data: {len(dates)} dates available")
        print(f"  Range: {dates[0]} to {dates[-1]}")
        print(f"  Data dir: {trade_dir}")

        sample_date = dates[len(dates) // 2]
        print(f"\n  Loading all trades for {sample_date} hour 12...")
        try:
            df_t = read_trades(trade_dir, date=sample_date, hour=12)
            print(f"  Loaded {len(df_t):,} trades across {df_t['coin'].nunique()} coins")
            print(f"\n  Top 5 coins by trade count:")
            for coin, count in df_t["coin"].value_counts().head(5).items():
                print(f"    {coin:15s}  {count:>6,}")

            # Filter demo
            print(f"\n  Loading BTC-only trades for {sample_date} hour 12...")
            df_btc = read_trades(trade_dir, date=sample_date, hour=12, coins=["BTC"])
            print(f"  {len(df_btc):,} BTC trades")
            print(f"  Price range: ${df_btc['px'].min():,.2f} – ${df_btc['px'].max():,.2f}")
        except ValueError as e:
            print(f"  {e}")
    else:
        print("\n  Trade data not found.")
        print("  Expected: directory with 'YYYYMMDD' subfolders containing .gz files")

    print()
    print("=" * 65)
    print("  See README.md for setup instructions and SCHEMA.md for details")
    print("=" * 65)
    print()
