import pandas as pd
from read_data import read_orders, read_trades
from collections import defaultdict

orders = read_orders('order_statuses', 'mapdir',
                     date='2025-12-01', hour=12, coin='eth')
orders = orders.sort_values('ts').reset_index(drop=True)
print(f"Loaded {len(orders):,} order events")

CANCEL_STATUSES = {'canceled', 'reduceOnlyCanceled', 'scheduledCancel',
                   'siblingFilledCanceled', 'selfTradeCanceled',
                   'marginCanceled', 'vaultWithdrawalCanceled', 'liquidatedCanceled'}
REJECT_STATUSES = {'badAloPxRejected', 'perpMarginRejected', 'iocCancelRejected',
                   'minTradeNtlRejected', 'reduceOnlyRejected',
                   'perpMaxPositionRejected', 'oracleRejected'}
RESTING_TIFS = {'Alo', 'Gtc'}

bids = defaultdict(dict)   # {price: {oid: size}}
asks = defaultdict(dict)
order_loc = {}

def best_bid():
    return max(bids.keys()) if bids else None
def best_ask():
    return min(asks.keys()) if asks else None

def match_incoming(is_ask, px, sz, incoming_oid):
    """Match an incoming order against the opposite side.
    Returns remaining unfilled size."""
    remaining = sz
    if not is_ask:
        # Incoming BUY matches against ASKS priced <= px, cheapest first
        while remaining > 0 and asks:
            best = best_ask()
            if best is None or best > px:
                break  # no longer crosses
            level = asks[best]
            for resting_oid in list(level.keys()):
                avail = level[resting_oid]
                traded = min(remaining, avail)
                remaining -= traded
                level[resting_oid] -= traded
                if level[resting_oid] <= 1e-9:
                    del level[resting_oid]
                    order_loc.pop(resting_oid, None)
                if remaining <= 1e-9:
                    break
            if not asks[best]:
                del asks[best]
    else:
        # Incoming SELL matches against BIDS priced >= px, highest first
        while remaining > 0 and bids:
            best = best_bid()
            if best is None or best < px:
                break
            level = bids[best]
            for resting_oid in list(level.keys()):
                avail = level[resting_oid]
                traded = min(remaining, avail)
                remaining -= traded
                level[resting_oid] -= traded
                if level[resting_oid] <= 1e-9:
                    del level[resting_oid]
                    order_loc.pop(resting_oid, None)
                if remaining <= 1e-9:
                    break
            if not bids[best]:
                del bids[best]
    return remaining

snapshots = []
rejected_count = 0

for row in orders.itertuples():
    oid, status, px, sz, is_ask = row.oid, row.status, row.limitPx, row.sz, row.isAsk
    otype, tif = row.orderType, row.tif

    if status == 'open':
        if otype == 'Limit' and tif in RESTING_TIFS:
            # First, match against the book (price-time priority)
            remaining = match_incoming(is_ask, px, sz, oid)
            # Whatever doesn't fill, rest it
            if remaining > 1e-9:
                side = asks if is_ask else bids
                side[px][oid] = remaining
                order_loc[oid] = (px, is_ask)

    elif status == 'filled':
        if oid in order_loc:
            p, a = order_loc[oid]
            side = asks if a else bids
            if sz <= 0:
                side[p].pop(oid, None)
                if p in side and not side[p]: del side[p]
                order_loc.pop(oid, None)
            else:
                side[p][oid] = sz

    elif status in CANCEL_STATUSES:
        if oid in order_loc:
            p, a = order_loc[oid]
            side = asks if a else bids
            side[p].pop(oid, None)
            if p in side and not side[p]: del side[p]
            order_loc.pop(oid, None)

    elif status in REJECT_STATUSES:
        rejected_count += 1

    if row.Index % 1000 == 0:
        bb, ba = best_bid(), best_ask()
        if bb and ba:
            snapshots.append({'ts': row.ts, 'bid': bb, 'ask': ba,
                              'spread': ba - bb, 'mid': (ba+bb)/2})

snap_df = pd.DataFrame(snapshots)
print(f"\nReconstructed {len(snap_df):,} snapshots")
print(f"Rejected orders: {rejected_count:,}")
print(f"\nSpread stats:")
print(snap_df['spread'].describe())
print(f"\nCrossed snapshots (spread<0): {(snap_df['spread']<0).sum():,} "
      f"({100*(snap_df['spread']<0).mean():.1f}%)")
print(f"\nSample:\n{snap_df.head(10)}")

trades = read_trades('trades', date='2025-12-01', hour=12, coins=['ETH'])
print(f"\nTrade price range: {trades['px'].min():.2f} – {trades['px'].max():.2f}")
print(f"Book mid range:    {snap_df['mid'].min():.2f} – {snap_df['mid'].max():.2f}")