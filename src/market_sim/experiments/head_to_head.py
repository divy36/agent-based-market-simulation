"""Original experiment: Maxime Nunes; packaged with statistical corrections."""
from ..statistics import holm_adjust
from ..market import DoubleAuctionMarket
from ..pipeline import SimulationPipeline
from ..agents import ZICAgent, ZIUAgent, GDAgent, RothErevAgent


def run(n_runs=50, n_ticks=1000):
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats

    N_PER_TYPE = 4
    N_RUNS = n_runs
    N_TICKS = n_ticks
    MIN_PRICE = 0
    MAX_PRICE = 200
    EQ_PRICE = 100
    COLOURS = {"ZI-C": "steelblue", "Roth-Erev": "darkorange", "GD": "mediumseagreen"}
    LABEL_MAP = {
        "ZIC": "ZI-C",
        "RothErev": "Roth-Erev",
        "GD": "GD",
        "ZICAgent": "ZI-C",
        "RothErevAgent": "Roth-Erev",
        "GDAgent": "GD",
    }

    def _ci(arr):
        a = [x for x in arr if x is not None and np.isfinite(x)]
        if len(a) < 2:
            return 0.0
        return 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))

    def _build_mixed(rng):
        """Build one set of 33/33/34 split buyers and sellers."""

        def agents(cls, is_buyer, start_id):
            (lo, hi) = (100, 200) if is_buyer else (0, 100)
            return [
                cls(start_id + i, is_buyer, rng.uniform(lo, hi), MIN_PRICE, MAX_PRICE)
                for i in range(N_PER_TYPE)
            ]

        buyers = (
            agents(ZICAgent, True, 0)
            + agents(RothErevAgent, True, 1000)
            + agents(GDAgent, True, 2000)
        )
        sellers = (
            agents(ZICAgent, False, 3000)
            + agents(RothErevAgent, False, 4000)
            + agents(GDAgent, False, 5000)
        )
        return (buyers, sellers)

    def _max_surplus_per_tick(buyers, sellers):
        bvals = sorted([b.valuation for b in buyers], reverse=True)
        scosts = sorted([s.valuation for s in sellers])
        return sum((max(0, v - c) for (v, c) in zip(bvals, scosts)))

    print("=" * 65)
    print("Experiment 3 - Head-to-Head: ZI-C vs Roth-Erev vs GD")
    print(f"  {N_PER_TYPE} agents per type per side | {N_RUNS} runs | {N_TICKS} ticks")
    print("=" * 65)
    run_records = []
    market_records = []
    example_trade_df = None
    example_tick_df = None
    for run in range(N_RUNS):
        rng = random.Random(run * 137 + 42)
        (buyers, sellers) = _build_mixed(rng)
        market = DoubleAuctionMarket(
            buyers, sellers, min_price=MIN_PRICE, max_price=MAX_PRICE, random_seed=run
        )
        market.run(N_TICKS)
        trade_df = market.get_trade_log()
        tick_df = market.get_tick_log()
        agent_df = market.get_agent_stats()
        if run == 0:
            example_trade_df = trade_df.copy()
            example_tick_df = tick_df.copy()
        total_surplus = agent_df["profit"].sum()
        max_surplus_pt = _max_surplus_per_tick(buyers, sellers)
        efficiency = min(
            100,
            max(
                0,
                (
                    total_surplus / (max_surplus_pt * N_TICKS) * 100
                    if max_surplus_pt > 0
                    else 0
                ),
            ),
        )
        market_records.append(
            {
                "run": run,
                "n_trades": len(trade_df),
                "avg_trade_price": (
                    trade_df["trade_price"].mean() if not trade_df.empty else np.nan
                ),
                "price_volatility": (
                    trade_df["trade_price"].std() if not trade_df.empty else np.nan
                ),
                "mean_spread": (
                    tick_df["spread"].dropna().mean() if not tick_df.empty else np.nan
                ),
                "total_surplus": total_surplus,
                "efficiency": efficiency,
            }
        )
        for _, row in agent_df.iterrows():
            label = LABEL_MAP.get(row["agent_type"], row["agent_type"])
            run_records.append(
                {
                    "run": run,
                    "agent_type": label,
                    "profit": row["profit"],
                    "trade_count": row["trade_count"],
                    "avg_profit_trade": row["avg_profit_per_trade"],
                    "surplus_share": (
                        row["profit"] / total_surplus if total_surplus > 0 else 0
                    ),
                }
            )
        if (run + 1) % 10 == 0:
            print(f"  Completed {run + 1}/{N_RUNS} runs...")
    run_df = pd.DataFrame(run_records)
    market_df = pd.DataFrame(market_records)

    def summarise(col):
        return (
            run_df.groupby(["run", "agent_type"])[col]
            .mean()
            .groupby("agent_type")
            .agg(mean="mean", ci=_ci)
            .reset_index()
        )

    profit_summary = summarise("profit")
    trade_summary = summarise("trade_count")
    apt_summary = summarise("avg_profit_trade")
    surplus_summary = (
        run_df.groupby(["run", "agent_type"])["surplus_share"]
        .sum()
        .groupby("agent_type")
        .agg(mean="mean", ci=_ci)
        .reset_index()
    )
    print("\n Agent-level summary (run-level means and 95% CI)")
    combined = profit_summary.copy()
    combined.columns = ["agent_type", "mean_profit", "profit_ci"]
    combined["mean_trades"] = trade_summary["mean"].values
    combined["trades_ci"] = trade_summary["ci"].values
    combined["mean_apt"] = apt_summary["mean"].values
    combined["apt_ci"] = apt_summary["ci"].values
    combined["surplus_share%"] = (surplus_summary["mean"] * 100).round(1).values
    print(combined.round(2).to_string(index=False))
    print("\n Market-level summary")
    for col in [
        "n_trades",
        "avg_trade_price",
        "price_volatility",
        "mean_spread",
        "efficiency",
    ]:
        vals = market_df[col].dropna()
        print(f"  {col:<22}: {vals.mean():.2f} ± {_ci(vals):.2f}")
    print("\n── Pairwise significance (one-sided paired t-test, Holm-corrected) ─")
    pairs = [("GD", "ZI-C"), ("GD", "Roth-Erev"), ("Roth-Erev", "ZI-C")]
    raw_results = []
    for a, b in pairs:
        a_vals = (
            run_df[run_df["agent_type"] == a].groupby("run")["profit"].mean().values
        )
        b_vals = (
            run_df[run_df["agent_type"] == b].groupby("run")["profit"].mean().values
        )
        (t, p2) = stats.ttest_rel(a_vals, b_vals)
        p1 = p2 / 2 if t > 0 else 1.0 - p2 / 2
        raw_results.append(
            {
                "pair": f"{a} > {b}",
                "t": t,
                "p_raw": p1,
                "gap": a_vals.mean() - b_vals.mean(),
            }
        )
    holm = holm_adjust([r["p_raw"] for r in raw_results])
    for i, r in enumerate(raw_results):
        sig = holm[i] < 0.05
        print(
            f"  {r['pair']:<25} gap={r['gap']:+.2f}  p_raw={r['p_raw']:.4f}  p_Holm={holm[i]:.4f}  {('✓ sig' if sig else '✗ not sig')}"
        )
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.32)
    fig.suptitle(
        "Experiment 3 - Head-to-Head: ZI-C vs Roth-Erev vs GD\nEqual agent-type split",
        fontsize=13,
        fontweight="bold",
    )
    types = ["ZI-C", "Roth-Erev", "GD"]
    colors = [COLOURS[t] for t in types]

    def _vals(df, col, types=types):
        out = []
        for t in types:
            rows = df[df["agent_type"] == t][col].values
            if len(rows) == 0:
                raise ValueError(
                    f"No rows found for agent_type='{t}' in column '{col}'. Available types: {df['agent_type'].unique().tolist()}"
                )
            out.append(rows[0])
        return out

    def _cis(df, col, types=types):
        return [df[df["agent_type"] == t]["ci"].values[0] for t in types]

    ax1 = fig.add_subplot(gs[0, 0])
    means = _vals(profit_summary, "mean")
    cis = _vals(profit_summary, "ci")
    bars = ax1.bar(
        types, means, yerr=cis, color=colors, alpha=0.85, capsize=6, width=0.5, zorder=3
    )
    for bar, val in zip(bars, means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cis) * 0.1,
            f"{val:.0f}",
            ha="center",
            fontsize=9,
        )
    ax1.set_title("Mean Profit per Agent by Type")
    ax1.set_ylabel("Profit (± 95% CI)")
    ax1.grid(axis="y", alpha=0.3)
    ax2 = fig.add_subplot(gs[0, 1])
    tmeans = _vals(trade_summary, "mean")
    tcis = _vals(trade_summary, "ci")
    bars2 = ax2.bar(
        types,
        tmeans,
        yerr=tcis,
        color=colors,
        alpha=0.85,
        capsize=6,
        width=0.5,
        zorder=3,
    )
    for bar, val in zip(bars2, tmeans):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tcis) * 0.1,
            f"{val:.1f}",
            ha="center",
            fontsize=9,
        )
    ax2.set_title("Mean Trade Count by Agent Type")
    ax2.set_ylabel("Trades per Agent (± 95% CI)")
    ax2.grid(axis="y", alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 0])
    apts = _vals(apt_summary, "mean")
    acis = _vals(apt_summary, "ci")
    bars3 = ax3.bar(
        types, apts, yerr=acis, color=colors, alpha=0.85, capsize=6, width=0.5, zorder=3
    )
    for bar, val in zip(bars3, apts):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(acis) * 0.1,
            f"{val:.2f}",
            ha="center",
            fontsize=9,
        )
    ax3.set_title("Avg Profit per Trade")
    ax3.set_ylabel("Profit / Trade (± 95% CI)")
    ax3.grid(axis="y", alpha=0.3)
    ax4 = fig.add_subplot(gs[1, 1])
    svals = [
        surplus_summary[surplus_summary["agent_type"] == t]["mean"].values[0] * 100
        for t in types
    ]
    scis = [
        surplus_summary[surplus_summary["agent_type"] == t]["ci"].values[0] * 100
        for t in types
    ]
    bars4 = ax4.bar(
        types,
        svals,
        yerr=scis,
        color=colors,
        alpha=0.85,
        capsize=6,
        width=0.5,
        zorder=3,
    )
    ax4.axhline(
        33.3, color="red", lw=1.2, ls="--", alpha=0.5, label="Equal share (33.3%)"
    )
    for bar, val in zip(bars4, svals):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center",
            fontsize=9,
        )
    ax4.set_title("Surplus Share by Agent Type")
    ax4.set_ylabel("% of Total Surplus (± 95% CI)")
    ax4.legend(fontsize=8, framealpha=0)
    ax4.grid(axis="y", alpha=0.3)
    ax5 = fig.add_subplot(gs[2, 0])
    if example_trade_df is not None and (not example_trade_df.empty):
        ma = example_trade_df["trade_price"].rolling(30).mean()
        ax5.scatter(
            example_trade_df["tick"],
            example_trade_df["trade_price"],
            s=5,
            alpha=0.2,
            color="steelblue",
        )
        ax5.plot(
            example_trade_df["tick"], ma, color="navy", lw=1.8, label="30-trade MA"
        )
    ax5.axhline(
        EQ_PRICE, color="red", lw=1.2, ls="--", label=f"Equilibrium ({EQ_PRICE})"
    )
    ax5.set_title("Price Convergence (representative run)")
    ax5.set_xlabel("Tick")
    ax5.set_ylabel("Trade Price")
    ax5.legend(fontsize=8, framealpha=0)
    ax6 = fig.add_subplot(gs[2, 1])
    profit_data = [
        run_df[run_df["agent_type"] == t].groupby("run")["profit"].sum().values
        for t in types
    ]
    parts = ax6.violinplot(
        profit_data, positions=[1, 2, 3], showmedians=True, showextrema=True
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    ax6.set_xticks([1, 2, 3])
    ax6.set_xticklabels(types)
    ax6.set_title(f"Profit Distribution Across {n_runs} Runs")
    ax6.set_ylabel("Total Profit per Run")
    ax6.grid(axis="y", alpha=0.3)
    return (market_df, run_df)
