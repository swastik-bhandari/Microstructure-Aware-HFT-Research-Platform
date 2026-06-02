from collections import defaultdict, deque
import numpy as np
import pandas as pd

from read_data import read_orders, read_trades


# ---------------------------------------------------------------------------
# Status categories (from SCHEMA.md statuses.csv)
# ---------------------------------------------------------------------------

CANCEL_STATUSES = {
    "canceled", "reduceOnlyCanceled", "scheduledCancel",
    "siblingFilledCanceled", "selfTradeCanceled", "marginCanceled",
    "vaultWithdrawalCanceled", "liquidatedCanceled",
}
REJECT_STATUSES = {
    "badAloPxRejected", "perpMarginRejected", "iocCancelRejected",
    "minTradeNtlRejected", "reduceOnlyRejected", "perpMaxPositionRejected",
    "oracleRejected",
}
# Only genuine resting limit orders sit in the book.
# IOC / Market / trigger orders never rest.
RESTING_TIFS = {"Alo", "Gtc"}

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Layer 1: Limit Order Book with price-time priority matching
# ---------------------------------------------------------------------------

class LOB:
    """Price-time-priority limit order book.

    Maintains bids and asks as {price: {oid: size}}. Insertion order of the
    inner dict preserves time priority (Python dicts are insertion-ordered).
    """

    def __init__(self):
        self.bids = defaultdict(dict)   # {price: {oid: size}}
        self.asks = defaultdict(dict)   # {price: {oid: size}}
        self.order_loc = {}             # {oid: (price, is_ask)}

        # Stats / signals
        self.rejected_count = 0
        self.fill_times = deque()       # timestamps (ns) of recent fills

    # ---- top of book -----------------------------------------------------

    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid(self):
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self):
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    # ---- matching --------------------------------------------------------

    def _match(self, is_ask, px, sz, ts):
        """Match an incoming order against the opposite side.
        Returns remaining (unfilled) size."""
        remaining = sz

        if not is_ask:
            # incoming BUY eats asks priced <= px, cheapest first
            while remaining > _EPS and self.asks:
                best = self.best_ask()
                if best is None or best > px:
                    break
                level = self.asks[best]
                for roid in list(level.keys()):       # time priority
                    avail = level[roid]
                    traded = min(remaining, avail)
                    remaining -= traded
                    level[roid] -= traded
                    if level[roid] <= _EPS:
                        del level[roid]
                        self.order_loc.pop(roid, None)
                    self.fill_times.append(ts)
                    if remaining <= _EPS:
                        break
                if not self.asks[best]:
                    del self.asks[best]
        else:
            # incoming SELL eats bids priced >= px, highest first
            while remaining > _EPS and self.bids:
                best = self.best_bid()
                if best is None or best < px:
                    break
                level = self.bids[best]
                for roid in list(level.keys()):
                    avail = level[roid]
                    traded = min(remaining, avail)
                    remaining -= traded
                    level[roid] -= traded
                    if level[roid] <= _EPS:
                        del level[roid]
                        self.order_loc.pop(roid, None)
                    self.fill_times.append(ts)
                    if remaining <= _EPS:
                        break
                if not self.bids[best]:
                    del self.bids[best]

        return remaining

    # ---- event processing -----------------------------------------------

    def process_event(self, oid, status, px, sz, is_ask, otype, tif, ts):
        """Apply a single order-status event to the book."""
        if status == "open":
            if otype == "Limit" and tif in RESTING_TIFS:
                remaining = self._match(is_ask, px, sz, ts)
                if remaining > _EPS:
                    side = self.asks if is_ask else self.bids
                    side[px][oid] = remaining
                    self.order_loc[oid] = (px, is_ask)

        elif status == "filled":
            if oid in self.order_loc:
                p, a = self.order_loc[oid]
                side = self.asks if a else self.bids
                if sz <= _EPS:                    # fully filled
                    side[p].pop(oid, None)
                    if p in side and not side[p]:
                        del side[p]
                    self.order_loc.pop(oid, None)
                else:                             # partial: sz = remaining
                    side[p][oid] = sz
                self.fill_times.append(ts)

        elif status in CANCEL_STATUSES:
            if oid in self.order_loc:
                p, a = self.order_loc[oid]
                side = self.asks if a else self.bids
                side[p].pop(oid, None)
                if p in side and not side[p]:
                    del side[p]
                self.order_loc.pop(oid, None)

        elif status in REJECT_STATUSES:
            self.rejected_count += 1

    # ---- depth / liquidity helpers --------------------------------------

    def level_volume(self, is_ask, price):
        side = self.asks if is_ask else self.bids
        return sum(side.get(price, {}).values())

    def top_n_levels(self, is_ask, n):
        """Return [(price, total_size), ...] for the best n levels."""
        side = self.asks if is_ask else self.bids
        if not side:
            return []
        prices = sorted(side.keys(), reverse=not is_ask)[:n]  # asks asc, bids desc
        return [(p, sum(side[p].values())) for p in prices]

    def queue_ahead(self, is_ask, price):
        """Approximate queue position: total resting size at `price`.

        A new order posted here would sit behind this much size (FIFO).
        Orders-only approximation, as committed in the proposal.
        """
        side = self.asks if is_ask else self.bids
        return sum(side.get(price, {}).values())


# ---------------------------------------------------------------------------
# Layer 2: Feature engine (time-sliced sampling)
# ---------------------------------------------------------------------------

class FeatureEngine:
    """Computes the 6-feature RL state vector from an LOB.

    Features:
        spread          : best_ask - best_bid
        obi             : bid_vol / (bid_vol + ask_vol) over top N levels  [0,1]
        depth           : total volume over top N levels (both sides)
        volatility      : rolling std of mid-price returns
        trade_intensity : EWMA count of fills per second
        queue_at_best   : resting size at the best bid (queue proxy)
    """

    def __init__(self, depth_levels=5, vol_window=20, intensity_halflife_s=5.0):
        self.depth_levels = depth_levels
        self.vol_window = vol_window
        self.intensity_halflife_s = intensity_halflife_s
        self.mid_history = deque(maxlen=vol_window + 1)
        self._last_fill_count = 0
        self._intensity = 0.0
        self._last_ts = None

    def _update_intensity(self, lob, ts):
        """EWMA of fills since last sample, normalized to per-second."""
        n_fills = len(lob.fill_times)
        new_fills = n_fills - self._last_fill_count
        self._last_fill_count = n_fills

        if self._last_ts is not None:
            dt = (ts - self._last_ts) / 1e9          # ns -> s
            if dt > 0:
                rate = new_fills / dt
                alpha = 1.0 - np.exp(-dt / self.intensity_halflife_s)
                self._intensity = (1 - alpha) * self._intensity + alpha * rate
        self._last_ts = ts
        return self._intensity

    def sample(self, lob, ts):
        """Return a feature dict, or None if the book isn't two-sided yet."""
        bb, ba = lob.best_bid(), lob.best_ask()
        if bb is None or ba is None:
            return None

        spread = ba - bb
        mid = (bb + ba) / 2.0

        bid_levels = lob.top_n_levels(is_ask=False, n=self.depth_levels)
        ask_levels = lob.top_n_levels(is_ask=True, n=self.depth_levels)
        bid_vol = sum(v for _, v in bid_levels)
        ask_vol = sum(v for _, v in ask_levels)
        total_vol = bid_vol + ask_vol
        obi = bid_vol / total_vol if total_vol > 0 else 0.5
        depth = total_vol

        # volatility: rolling std of log returns of mid
        self.mid_history.append(mid)
        if len(self.mid_history) >= 3:
            arr = np.array(self.mid_history)
            rets = np.diff(np.log(arr))
            vol = float(np.std(rets))
        else:
            vol = 0.0

        intensity = self._update_intensity(lob, ts)
        queue_at_best = lob.queue_ahead(is_ask=False, price=bb)

        return {
            "ts": ts,
            "mid": mid,
            "spread": spread,
            "obi": obi,
            "depth": depth,
            "volatility": vol,
            "trade_intensity": intensity,
            "queue_at_best": queue_at_best,
        }


# ---------------------------------------------------------------------------
# Driver: replay events, sample features on a fixed time grid
# ---------------------------------------------------------------------------

def build_states(date="2025-12-01", hour=12, coin="eth",
                 interval_ms=1000, depth_levels=5,
                 order_dir="order_statuses", map_dir="mapdir"):
    """Replay one hour of order events and emit a time-sliced state table.

    Returns a DataFrame, one row per sampling interval, with the 6 features.
    """
    orders = read_orders(order_dir, map_dir, date=date, hour=hour, coin=coin)
    orders = orders.sort_values("ts").reset_index(drop=True)

    # ns timestamps as plain int64 for arithmetic
    ts_ns = orders["ts"].astype("int64").to_numpy()

    lob = LOB()
    feat = FeatureEngine(depth_levels=depth_levels)

    interval_ns = interval_ms * 1_000_000
    next_sample = ts_ns[0] + interval_ns
    states = []

    oid_a = orders["oid"].to_numpy()
    status_a = orders["status"].to_numpy()
    px_a = orders["limitPx"].to_numpy()
    sz_a = orders["sz"].to_numpy()
    isask_a = orders["isAsk"].to_numpy()
    otype_a = orders["orderType"].to_numpy()
    tif_a = orders["tif"].to_numpy()

    for i in range(len(orders)):
        t = ts_ns[i]
        # sample at every interval boundary we've crossed
        while t >= next_sample:
            s = feat.sample(lob, next_sample)
            if s is not None:
                states.append(s)
            next_sample += interval_ns

        lob.process_event(
            oid_a[i], status_a[i], px_a[i], sz_a[i],
            isask_a[i], otype_a[i], tif_a[i], t,
        )

    return pd.DataFrame(states), lob


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building Layer 2 states for ETH 2025-12-01 hour 12 ...")
    states, lob = build_states(date="2025-12-01", hour=12, interval_ms=1000)

    print(f"\nGenerated {len(states):,} state vectors (1s interval)")
    print(f"Rejected orders this hour: {lob.rejected_count:,}")

    print("\n=== Feature sanity checks ===")
    print(f"spread > 0 always:      {(states['spread'] > 0).all()}")
    print(f"obi in [0,1] always:    {states['obi'].between(0,1).all()}")
    print(f"depth >= 0 always:      {(states['depth'] >= 0).all()}")
    print(f"volatility >= 0 always: {(states['volatility'] >= 0).all()}")

    print("\n=== Feature distributions ===")
    print(states[["spread", "obi", "depth", "volatility",
                  "trade_intensity", "queue_at_best"]].describe())

    print("\n=== Sample state vectors ===")
    print(states.head(10).to_string())

    # cross-check mid against real trades
    trades = read_trades("trades", date="2025-12-01", hour=12, coins=["ETH"])
    print(f"\nTrade price range: {trades['px'].min():.2f} - {trades['px'].max():.2f}")
    print(f"State mid range:   {states['mid'].min():.2f} - {states['mid'].max():.2f}")