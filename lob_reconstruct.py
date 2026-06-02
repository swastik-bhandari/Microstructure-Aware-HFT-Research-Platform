import pandas as pd
from read_data import read_orders, read_trades
from collections import defaultdict

# --- Load one hour of ETH orders ---
orders = read_orders('order_statuses', 'mapdir',
                     date='2025-12-01', hour=12, coin='eth')
orders = orders.sort_values('ts').reset_index(drop=True)
print(f"Loaded {len(orders):,} order events")
print(f"Status breakdown:\n{orders['status'].value_counts()}\n")

# --- Status categories ---
CANCEL_STATUSES = {'canceled', 'reduceOnlyCanceled', 'scheduledCancel',
                   'siblingFilledCanceled', 'selfTradeCanceled',
                   'marginCanceled', 'vaultWithdrawalCanceled', 'liquidatedCanceled'}
REJECT_STATUSES = {'badAloPxRejected', 'perpMarginRejected', 'iocCancelRejected',
                   'minTradeNtlRejected', 'reduceOnlyRejected',
                   'perpMaxPositionRejected', 'oracleRejected'}

# --- Book state ---
bids = defaultdict(dict)  # {price: {oid: size}}
asks = defaultdict(dict)
order_loc = {}            # {oid: (price, is_ask)}

def best_bid():
    return max(bids.keys()) if bids else None

def best_ask():
    return min(asks.keys()) if asks else None

# --- Replay ---
snapshots = []
rejected_count = 0

for row in orders.itertuples():
    oid, status, px, sz, is_ask = row.oid, row.status, row.limitPx, row.sz, row.isAsk

    if status == 'open':
        side = asks if is_ask else bids
        side[px][oid] = sz
        order_loc[oid] = (px, is_ask)

    elif status == 'filled':
        if oid in order_loc:
            p, a = order_loc[oid]
            side = asks if a else bids
            if sz <= 0:                      # fully filled
                side[p].pop(oid, None)
                if not side[p]: del side[p]
                order_loc.pop(oid, None)
            else:                            # partial — update remaining
                side[p][oid] = sz

    elif status in CANCEL_STATUSES:
        if oid in order_loc:
            p, a = order_loc[oid]
            side = asks if a else bids
            side[p].pop(oid, None)
            if p in side and not side[p]: del side[p]
            order_loc.pop(oid, None)

    elif status in REJECT_STATUSES:
        rejected_count += 1               # never touches book — your unique signal

    # snapshot top-of-book every 1000 events
    if row.Index % 1000 == 0:
        bb, ba = best_bid(), best_ask()
        if bb and ba:
            snapshots.append({'ts': row.ts, 'bid': bb, 'ask': ba,
                              'spread': ba - bb, 'mid': (ba+bb)/2})

snap_df = pd.DataFrame(snapshots)
print(f"Reconstructed {len(snap_df):,} book snapshots")
print(f"Rejected orders this hour: {rejected_count:,}")
print(f"\nSpread stats (should be small & positive for ETH):")
print(snap_df['spread'].describe())
print(f"\nSample snapshots:\n{snap_df.head(10)}")

# --- Validate against trades ---
trades = read_trades('trades', date='2025-12-01', hour=12, coins=['ETH'])
print(f"\nETH trades this hour: {len(trades):,}")
print(f"Trade price range: {trades['px'].min():.2f} – {trades['px'].max():.2f}")
print(f"Book mid range:    {snap_df['mid'].min():.2f} – {snap_df['mid'].max():.2f}")
print("(These two ranges should overlap heavily if reconstruction is correct)")