import pandas as pd
from read_data import read_orders
from collections import defaultdict

orders = read_orders('order_statuses', 'mapdir',
                     date='2025-12-01', hour=12, coin='eth')
orders = orders.sort_values('ts').reset_index(drop=True)

# Q1: What order types are these "open" events?
print("=== Order types among 'open' events ===")
opens = orders[orders['status'] == 'open']
print(opens['orderType'].value_counts())
print()

# Q2: What TIF? (Alo=rests, Ioc=doesn't rest, Gtc=rests)
print("=== TIF among 'open' events ===")
print(opens['tif'].value_counts())
print()

# Q3: Do cancels/fills reference oids we've seen as 'open'?
print("=== Do cancel/fill oids match open oids? ===")
open_oids = set(orders[orders['status'] == 'open']['oid'])
cancel_oids = set(orders[orders['status'] == 'canceled']['oid'])
fill_oids = set(orders[orders['status'] == 'filled']['oid'])
print(f"Open oids:   {len(open_oids):,}")
print(f"Cancel oids: {len(cancel_oids):,}")
print(f"  of which match an open: {len(cancel_oids & open_oids):,} "
      f"({100*len(cancel_oids & open_oids)/max(len(cancel_oids),1):.1f}%)")
print(f"Fill oids:   {len(fill_oids):,}")
print(f"  of which match an open: {len(fill_oids & open_oids):,} "
      f"({100*len(fill_oids & open_oids)/max(len(fill_oids),1):.1f}%)")
print()

# Q4: Price distribution of 'open' orders — are there crazy prices?
print("=== Price distribution of 'open' limit orders ===")
print(opens['limitPx'].describe())
print()

# Q5: For one specific oid that opens AND cancels, show its full lifecycle
print("=== Sample order lifecycle (one oid, all events) ===")
sample_oid = list(cancel_oids & open_oids)[0]
life = orders[orders['oid'] == sample_oid][
    ['ts','status','orderType','tif','isAsk','limitPx','sz','origSz']]
print(life.to_string())