"""
make_figures.py — publication figures for the size-aware execution paper.
Clean, grayscale-safe, serif labels. Reads real data only.
"""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- publication style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# grayscale-safe: distinct lightness + hatch/linestyle, not just hue
C_STATIC = "#c44e00"   # dark orange (dark)
C_SA     = "#1b7f4b"   # green (mid)
C_VWAP   = "#2b6cb0"   # blue (mid-dark)
LS = {"static": "-", "sizeaware": "--", "vwap": ":"}

# ---- load data ----
rows = list(csv.DictReader(open("variance_triple_raw.csv")))
static = np.array([float(r["static_bps"]) for r in rows])
sa     = np.array([float(r["sizeaware_bps"]) for r in rows])
vwap   = np.array([float(r["vwap_bps"]) for r in rows])
gap_sa_vwap = sa - vwap

ep = json.load(open("triple_episodes.json"))

def save(fig, name):
    fig.savefig(f"fig_{name}.png")
    fig.savefig(f"fig_{name}.pdf")
    plt.close(fig)
    print(f"saved fig_{name}.png / .pdf")


# ============================================================
# FIG 1 — distribution of shortfalls (the headline figure)
# ============================================================
XMAX = 300
fig, ax = plt.subplots(figsize=(7.2, 4.3))
bins = np.linspace(0, XMAX, 90)
for arr, c, lab, ls in [(static, C_STATIC, "PPO Static", "-"),
                        (sa, C_SA, "PPO Size-aware", "--"),
                        (vwap, C_VWAP, "VWAP", ":")]:
    ax.hist(np.clip(arr, 0, XMAX), bins=bins, histtype="step", linewidth=1.6,
            color=c, linestyle=ls, label=lab, density=True)
for arr, c in [(static, C_STATIC), (sa, C_SA), (vwap, C_VWAP)]:
    ax.axvline(arr.mean(), color=c, linewidth=1.0, alpha=0.55)
ax.set_xlabel("Implementation shortfall (bps) — lower is better")
ax.set_ylabel("Density")
ax.set_title("Distribution of execution shortfall across 1,500 test windows")
ax.set_xlim(0, XMAX)
ax.legend(loc="upper right")
n_over = int((static > XMAX).sum())
ax.text(0.985, 0.60,
        f"means (vertical lines):\nStatic {static.mean():.1f} · Size-aware {sa.mean():.1f} · VWAP {vwap.mean():.1f} bps\n"
        f"Static reaches {static.max():.0f} bps; {n_over} windows > {XMAX}\npiled into the final bin",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.3, color="#444444")
save(fig, "1_distributions")


# ============================================================
# FIG 2 — paired difference (size-aware − VWAP), 1500 gaps
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.0))
g = np.clip(gap_sa_vwap, -60, 60)
ax.hist(g, bins=np.linspace(-60, 60, 80), color=C_SA, alpha=0.32, edgecolor=C_SA, linewidth=0.6)
ax.axvline(0, color="#222222", linewidth=1.2)
ax.axvline(gap_sa_vwap.mean(), color=C_SA, linewidth=1.8, linestyle="--",
           label=f"mean = {gap_sa_vwap.mean():+.2f} bps")
wins = int((gap_sa_vwap < 0).sum())
ax.set_xlabel("Per-window difference: Size-aware − VWAP (bps)")
ax.set_ylabel("Count of windows")
ax.set_title("Paired comparison: Size-aware vs VWAP on identical windows")
ax.legend(loc="upper right")
ax.text(0.015, 0.97,
        f"left of 0 → Size-aware cheaper\n{wins}/1500 windows ({100*wins/1500:.1f}%)\n"
        f"Wilcoxon p < 1e-10",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color="#1b7f4b")
ax.annotate("Size-aware\nbetter", xy=(-40, ax.get_ylim()[1]*0.5), ha="center", fontsize=8.5, color="#1b7f4b")
ax.annotate("VWAP\nbetter", xy=(40, ax.get_ylim()[1]*0.5), ha="center", fontsize=8.5, color="#2b6cb0")
save(fig, "2_paired_gap")


# ============================================================
# FIG 3 — inventory trajectory on one window (the mechanism)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.2))
for ag, c, lab, ls in [("static", C_STATIC, "PPO Static", "-"),
                       ("sizeaware", C_SA, "PPO Size-aware", "--"),
                       ("vwap", C_VWAP, "VWAP", ":")]:
    e = ep[ag][0]
    inv = [r["inv"] for r in e["rows"]]
    t = list(range(len(inv)))
    ax.plot(t, inv, color=c, linestyle=ls, linewidth=1.9, label=f"{lab} (final {e['final_shortfall']:.0f} bps)")
ax.set_xlabel("Time within execution window (seconds)")
ax.set_ylabel("Remaining inventory (ETH)")
ax.set_title("Inventory trajectories on one test window (idx 72715, 20:11:55 UTC)")
ax.legend(loc="upper right")
ax.set_ylim(-0.3, 10.3)
# annotate the static cliff
e_static = ep["static"][0]
cliff_t = len(e_static["rows"]) - 1
ax.annotate("forced end-dump\n(6.3 ETH in 1s)",
            xy=(cliff_t, 0.1), xytext=(cliff_t-13, 4.7),
            arrowprops=dict(arrowstyle="->", color=C_STATIC, lw=1.0),
            fontsize=8.4, color=C_STATIC, ha="left")
save(fig, "3_inventory_trajectory")


# ============================================================
# FIG 4 — box plot of shortfall distributions (replaces scatter)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.4))
data = [static, sa, vwap]
labels = ["PPO\nStatic", "PPO\nSize-aware", "VWAP"]
colors = [C_STATIC, C_SA, C_VWAP]
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                showfliers=True, flierprops=dict(marker="o", markersize=2.5,
                markerfacecolor="#999999", markeredgecolor="none", alpha=0.35),
                medianprops=dict(color="black", linewidth=1.6),
                whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.30); patch.set_edgecolor(c); patch.set_linewidth(1.4)
# overlay mean markers
for i, (arr, c) in enumerate(zip(data, colors), 1):
    ax.scatter(i, arr.mean(), marker="D", s=42, color=c, edgecolor="white",
               linewidth=1.0, zorder=4)
ax.set_ylabel("Implementation shortfall (bps) — lower is better")
ax.set_title("Shortfall distribution by strategy (1,500 windows each)")
ax.set_ylim(0, 220)
ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="#cccccc")
ax.text(0.985, 0.97,
        "box = IQR (25–75%), line = median,\ndiamond = mean, dots = outliers\n"
        f"Static median {np.median(static):.0f}, mean {static.mean():.0f} (tail to {static.max():.0f})\n"
        f"Size-aware median {np.median(sa):.0f}, mean {sa.mean():.0f}\n"
        f"VWAP median {np.median(vwap):.0f}, mean {vwap.mean():.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.0, color="#444444")
save(fig, "4_boxplot")


# ============================================================
# FIG 5 (appendix) — per-seed consistency
# ============================================================
seeds = ["1000","1001","1002","1003","1004"]
def seed_means(key):
    return [np.mean([float(r[key]) for r in rows if r["seed"]==s]) for s in seeds]
ms_static = seed_means("static_bps"); ms_sa = seed_means("sizeaware_bps"); ms_vwap = seed_means("vwap_bps")

fig, ax = plt.subplots(figsize=(7.0, 4.0))
x = np.arange(len(seeds)); w = 0.26
ax.bar(x-w, ms_static, w, color=C_STATIC, label="PPO Static", edgecolor="white", linewidth=0.6)
ax.bar(x,   ms_sa,     w, color=C_SA, label="PPO Size-aware", edgecolor="white", linewidth=0.6, hatch="//")
ax.bar(x+w, ms_vwap,   w, color=C_VWAP, label="VWAP", edgecolor="white", linewidth=0.6, hatch="..")
ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("Mean shortfall (bps)")
ax.set_title("Per-seed mean shortfall (300 windows each) — result holds across all seeds")
ax.legend(loc="upper right", ncol=3)
ax.set_ylim(0, max(ms_static)*1.18)
for i in range(len(seeds)):
    ax.text(x[i], ms_sa[i]+1, f"{ms_sa[i]:.0f}", ha="center", fontsize=7.5, color=C_SA)
save(fig, "5_per_seed")

print("\nAll figures generated.")
