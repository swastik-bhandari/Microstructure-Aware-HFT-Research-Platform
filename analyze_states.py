
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load(file):
    df = pd.read_csv(file)
    # ts is int64 ns since epoch -> datetime for plotting
    df["dt"] = pd.to_datetime(df["ts"], unit="ns")
    return df


def plot_overview(df, out_png):
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    axes[0].plot(df["dt"], df["mid"], lw=0.6, color="black")
    axes[0].set_ylabel("Mid ($)")
    axes[0].set_title("ETH — Full Day Microstructure (1s sampling)")

    axes[1].plot(df["dt"], df["spread"], lw=0.4, color="tab:red")
    axes[1].set_ylabel("Spread ($)")

    axes[2].plot(df["dt"], df["obi"], lw=0.4, color="tab:blue")
    axes[2].axhline(0.5, color="grey", ls="--", lw=0.6)
    axes[2].set_ylabel("OBI")
    axes[2].set_ylim(0, 1)

    axes[3].plot(df["dt"], df["depth"], lw=0.4, color="tab:green")
    axes[3].set_ylabel("Depth (top 5)")

    axes[4].plot(df["dt"], df["trade_intensity"], lw=0.4, color="tab:orange")
    axes[4].set_ylabel("Trade intensity\n(fills/s)")

    axes[5].plot(df["dt"], df["volatility"], lw=0.4, color="tab:purple")
    axes[5].set_ylabel("Volatility")
    axes[5].set_xlabel("Time (UTC)")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"Saved figure -> {out_png}")


def analyze(df):
    print("\n" + "=" * 60)
    print(" FULL-DAY ANALYSIS")
    print("=" * 60)

    print(f"\nTotal states: {len(df):,}")
    print(f"Time span:    {df['dt'].min()}  ->  {df['dt'].max()}")
    print(f"Mid range:    ${df['mid'].min():.2f} - ${df['mid'].max():.2f} "
          f"({100*(df['mid'].max()/df['mid'].min()-1):.1f}% intraday move)")

    print("\n--- Hourly summary ---")
    hourly = df.groupby("hour").agg(
        states=("mid", "size"),
        mid_mean=("mid", "mean"),
        spread_mean=("spread", "mean"),
        obi_mean=("obi", "mean"),
        depth_mean=("depth", "mean"),
        intensity_mean=("trade_intensity", "mean"),
        vol_mean=("volatility", "mean"),
    ).round(3)
    print(hourly.to_string())

    print("\n--- Feature correlations with |mid return| (1s) ---")
    df = df.sort_values("ts")
    df["mid_ret"] = df["mid"].pct_change().abs()
    cols = ["spread", "obi", "depth", "trade_intensity", "volatility",
            "queue_at_best"]
    corr = df[cols + ["mid_ret"]].corr()["mid_ret"].drop("mid_ret").round(3)
    print(corr.to_string())
    print("\n(Higher |corr| = feature carries more signal about price moves.)")

    print("\n--- Cross-feature correlations ---")
    print(df[cols].corr().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="states_eth_20251201.csv")
    args = ap.parse_args()

    df = load(args.file)
    out_png = args.file.replace(".csv", "_overview.png")
    plot_overview(df, out_png)
    analyze(df)
    print("\nDone. Open the PNG to eyeball the day.")