"""
inject_test_episodes.py
=======================
Swap real out-of-sample test episodes (from record_test_episodes.py) into the
PPO agent sim inside showcase.html.

It replaces the embedded `const PPO_EP = [...]` block AND the `const names=[...]`
tab labels. Labels are NEUTRAL and DATA-DERIVED -- no hand-picked "calm/selloff"
naming. Each tab is named by its test-set start index and the realized 60s
direction of price (computed from the episode itself), e.g.

    "Test · idx 12236 · drift +0.4%"

so the label is earned from the data, not asserted.

Usage:
    python record_test_episodes.py --model models/ppo7_seed1000.zip --n 8 --seed 1
    python inject_test_episodes.py            # rewrites showcase.html in place
    python inject_test_episodes.py --episodes test_episodes.json --html showcase.html
"""
import argparse, json, re


def make_label(ep):
    """Neutral, data-derived tab label for one episode."""
    rows = ep["rows"]
    first_mid = rows[0]["mid"]
    last_mid = rows[-1]["mid"]
    drift = (last_mid - first_mid) / first_mid * 100.0   # % move over the 60s window
    idx = ep.get("start_idx", ep.get("start", "?"))      # absolute row in full day
    clock = rows[0].get("t", "")                          # human clock time if present
    when = f" {clock}" if clock and ":" in str(clock) else ""
    return f"Test idx {idx}{when} ({drift:+.2f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="test_episodes.json")
    ap.add_argument("--html", default="showcase.html")
    ap.add_argument("--max", type=int, default=6,
                    help="cap how many episodes get embedded (keeps file small / tabs readable)")
    args = ap.parse_args()

    eps = json.load(open(args.episodes))
    if args.max and len(eps) > args.max:
        eps = eps[:args.max]
        print(f"capping to first {args.max} episodes (use --max to change)")

    # the sim's JS reads r.action/r.sold/r.inv/r.pup/r.shortfall/r.mid/r.t and e.start/e.arrival.
    # record_test_episodes.py already produces those fields; normalise the couple of name diffs.
    for e in eps:
        e.setdefault("start", e.get("start_idx", 0))
        e.setdefault("final_sf", e.get("final_shortfall", 0))
        for r in e["rows"]:
            r.setdefault("pup", 0.5)
            if r.get("pup") is None:
                r["pup"] = 0.5

    names = [make_label(e) for e in eps]

    import os
    if not os.path.exists(args.html):
        raise SystemExit(
            f"\nERROR: '{args.html}' not found in this folder.\n"
            f"Either copy your showcase.html here, or pass its path, e.g.:\n"
            f'  python inject_test_episodes.py --html "C:\\Users\\ACER\\Downloads\\showcase.html"\n')
    html = open(args.html, encoding="utf-8").read()

    # --- replace the PPO_EP block (lambda repl avoids regex-escape issues in JSON) ---
    ep_json = json.dumps(eps, separators=(",", ":"))
    new_ep = "const PPO_EP = " + ep_json + ";"
    html, n1 = re.subn(r'const PPO_EP = \[.*?\}\];', lambda m: new_ep, html, count=1, flags=re.S)
    assert n1 == 1, "could not find the PPO_EP block to replace"

    # --- replace the names array (tab labels); ensure_ascii keeps it regex-safe ---
    # The agent sim declares its labels inline as:  const EP=PPO_EP, names=[...];
    # That is the one that builds the #aeptabs tabs, so target it specifically.
    names_js = json.dumps(names, ensure_ascii=True)
    html, n2 = re.subn(r'(const EP=PPO_EP,\s*names=)\[[^\]]*\]',
                       lambda m: m.group(1) + names_js, html, flags=re.S)
    if n2 == 0:
        # fallback: replace any standalone names array
        html, n2 = re.subn(r'const names=\[[^\]]*\];',
                           lambda m: "const names=" + names_js + ";", html, flags=re.S)
    print(f"replaced PPO_EP ({n1}x) and names array ({n2}x)")

    open(args.html, "w", encoding="utf-8").write(html)
    print(f"\nembedded {len(eps)} REAL test episodes into {args.html}")
    print("tab labels:")
    for nm in names:
        print("  -", nm)


if __name__ == "__main__":
    main()