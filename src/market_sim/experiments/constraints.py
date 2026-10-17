"""Original experiment: Divy Chauhan; packaged with statistical corrections."""
from ..statistics import holm_adjust
from ..market import DoubleAuctionMarket
from ..pipeline import SimulationPipeline
from ..agents import ZICAgent, ZIUAgent, GDAgent, RothErevAgent


def run(n_runs=30, n_ticks=300, sizes=(100, 500, 1000)):
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        from scipy import stats
    except Exception:
        stats = None

    def _max_surplus_per_tick(buyers, sellers) -> float:
        """Theoretical one-tick surplus under efficient matching."""
        buyer_vals = sorted([b.valuation for b in buyers], reverse=True)
        seller_costs = sorted([s.valuation for s in sellers])
        max_surplus = 0.0
        for v, c in zip(buyer_vals, seller_costs):
            if v >= c:
                max_surplus += v - c
            else:
                break
        return max_surplus

    def _time_normalized_efficiency(market, n_ticks: int) -> float:
        """Efficiency over the full horizon, clipped to [0, 100]."""
        per_tick = _max_surplus_per_tick(market.buyers, market.sellers)
        if per_tick <= 0 or n_ticks <= 0:
            return 0.0
        realised = sum((a.profit for a in market.all_agents))
        eff = realised / (per_tick * n_ticks) * 100.0
        return max(0.0, min(100.0, eff))

    def _trade_conditional_efficiency(market, tick_df) -> float:
        """Alternative efficiency normalised by ticks that had at least one trade."""
        per_tick = _max_surplus_per_tick(market.buyers, market.sellers)
        if per_tick <= 0:
            return 0.0
        if tick_df.empty or "n_trades" not in tick_df.columns:
            return 0.0
        ticks_with_trades = int((tick_df["n_trades"] > 0).sum())
        if ticks_with_trades == 0:
            return 0.0
        realised = sum((a.profit for a in market.all_agents))
        eff = realised / (per_tick * ticks_with_trades) * 100.0
        return max(0.0, min(100.0, eff))

    def run_experiment1(N_values=sizes, n_runs=n_runs, n_ticks=n_ticks):
        """Run Experiment 1: ZI-C vs ZI-U across population sizes."""
        all_rows = []
        run_rows = []

        def ci(arr):
            clean = [x for x in arr if x is not None and (not np.isnan(x))]
            if len(clean) < 2:
                return 0.0
            return 1.96 * np.std(clean, ddof=1) / np.sqrt(len(clean))

        for AgentClass, label in [(ZICAgent, "ZI-C"), (ZIUAgent, "ZI-U")]:
            for N in N_values:
                print(f"  Running {label}, N={N} ({n_runs} runs)...", end=" ")
                (efficiencies, spreads, volatilities, trade_prices) = ([], [], [], [])
                cond_efficiencies = []
                for run in range(n_runs):
                    rng = random.Random(run)
                    buyers = [
                        AgentClass(i, True, rng.uniform(60, 140), 1, 200)
                        for i in range(N // 2)
                    ]
                    sellers = [
                        AgentClass(N // 2 + i, False, rng.uniform(60, 140), 1, 200)
                        for i in range(N // 2)
                    ]
                    market = DoubleAuctionMarket(
                        buyers, sellers, min_price=1, max_price=200, random_seed=run
                    )
                    market.run(n_ticks)
                    pipe = SimulationPipeline(
                        market, experiment_id=f"{label}_N{N}_run{run}"
                    )
                    eff = _time_normalized_efficiency(market, n_ticks)
                    efficiencies.append(eff)
                    tick_df = pipe.ticks
                    cond_eff = _trade_conditional_efficiency(market, tick_df)
                    cond_efficiencies.append(cond_eff)
                    run_rows.append(
                        {
                            "Agent": label,
                            "N": N,
                            "run": run,
                            "Efficiency (%)": eff,
                            "Cond. Efficiency (%)": cond_eff,
                        }
                    )
                    if not tick_df.empty:
                        spreads.append(tick_df["spread"].dropna().mean())
                        volatilities.append(tick_df["mid_price"].dropna().std())
                    trade_df = pipe.trades
                    if not trade_df.empty:
                        trade_prices.append(trade_df["trade_price"].mean())
                all_rows.append(
                    {
                        "Agent": label,
                        "N": N,
                        "Efficiency (%)": round(np.mean(efficiencies), 2),
                        "Efficiency CI": round(ci(efficiencies), 2),
                        "Cond. Efficiency (%)": round(np.mean(cond_efficiencies), 2),
                        "Cond. Efficiency CI": round(ci(cond_efficiencies), 2),
                        "Mean Spread": (
                            round(np.mean(spreads), 2) if spreads else np.nan
                        ),
                        "Spread CI": round(ci(spreads), 2) if spreads else np.nan,
                        "Volatility": (
                            round(np.mean(volatilities), 2) if volatilities else np.nan
                        ),
                        "Volatility CI": (
                            round(ci(volatilities), 2) if volatilities else np.nan
                        ),
                        "Mean Trade Price": (
                            round(np.mean(trade_prices), 2) if trade_prices else np.nan
                        ),
                    }
                )
                print(f"efficiency={all_rows[-1]['Efficiency (%)']:.1f}%  ✓")
        return (pd.DataFrame(all_rows), pd.DataFrame(run_rows))

    print("=" * 60)
    print("Experiment 1: ZI-C vs ZI-U - Effect of N & Constraint")
    print("=" * 60)
    (df_results, df_eff_runs) = run_experiment1(
        N_values=sizes, n_runs=n_runs, n_ticks=n_ticks
    )
    print("\nResults Table ")
    print(
        df_results[
            [
                "Agent",
                "N",
                "Efficiency (%)",
                "Efficiency CI",
                "Volatility",
                "Volatility CI",
                "Mean Spread",
                "Spread CI",
            ]
        ].to_string(index=False)
    )
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Experiment 1 (Divy Chauhan): ZI-C vs ZI-U - Effect of Population Size\nBased on Gode & Sunder (1993)",
        fontsize=12,
        fontweight="bold",
    )
    colours = {"ZI-C": "steelblue", "ZI-U": "tomato"}
    for label, colour in colours.items():
        sub = df_results[df_results["Agent"] == label].sort_values("N")
        axes[0].errorbar(
            sub["N"],
            sub["Efficiency (%)"],
            yerr=sub["Efficiency CI"],
            fmt="o-",
            color=colour,
            lw=2,
            capsize=5,
            label=label,
        )
        axes[1].errorbar(
            sub["N"],
            sub["Volatility"],
            yerr=sub["Volatility CI"],
            fmt="o-",
            color=colour,
            lw=2,
            capsize=5,
            label=label,
        )
        axes[2].errorbar(
            sub["N"],
            sub["Mean Spread"],
            yerr=sub["Spread CI"],
            fmt="o-",
            color=colour,
            lw=2,
            capsize=5,
            label=label,
        )
    axes[0].set(title="Allocative Efficiency vs N", xlabel="N", ylabel="Efficiency (%)")
    axes[0].axhline(100, color="green", linestyle="--", alpha=0.4, label="Max (100%)")
    axes[0].set_ylim(0, 110)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set(
        title="Price Volatility vs N", xlabel="N", ylabel="Std Dev of Mid-Price"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].set(title="Bid-Ask Spread vs N", xlabel="N", ylabel="Mean Spread")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    _wide_all = df_eff_runs.pivot_table(
        index=["N", "run"], columns="Agent", values="Efficiency (%)", aggfunc="first"
    ).dropna(subset=["ZI-C", "ZI-U"])
    _wide_all = _wide_all.groupby(level="run").mean()
    zic_eff_vals = _wide_all["ZI-C"].to_numpy()
    ziu_eff_vals = _wide_all["ZI-U"].to_numpy()
    zic_eff = zic_eff_vals.mean() if len(zic_eff_vals) else np.nan
    ziu_eff = ziu_eff_vals.mean() if len(ziu_eff_vals) else np.nan
    paired_gaps = zic_eff_vals - ziu_eff_vals
    gap = paired_gaps.mean() if len(paired_gaps) else np.nan
    gap_ci = (
        1.96 * paired_gaps.std(ddof=1) / np.sqrt(len(paired_gaps))
        if len(paired_gaps) > 1
        else np.nan
    )
    if stats is not None and len(_wide_all) > 1:
        (t_stat, p_val_two) = stats.ttest_rel(
            _wide_all["ZI-C"], _wide_all["ZI-U"], nan_policy="omit"
        )
        p_val = p_val_two / 2 if t_stat > 0 else 1.0 - p_val_two / 2
    else:
        (t_stat, p_val) = (np.nan, np.nan)
    print("\nKey Finding")
    print(f"  ZI-C avg efficiency       : {zic_eff:.1f}%")
    print(f"  ZI-U avg efficiency       : {ziu_eff:.1f}%")
    print(f"  Mean paired gap           : {gap:.2f} pp")
    print(f"  Gap 95% CI                : ±{gap_ci:.2f} pp")
    print(f"  One-sided t-statistic     : {t_stat:.3f}")
    print(f"  One-sided p-value         : {p_val:.4f}  (H1: ZI-C > ZI-U)")
    print("\n  Per-N significance (one-sided paired t-test, Holm-corrected):")
    N_vals = sorted(df_eff_runs["N"].unique())
    raw_stats = []
    for N in N_vals:
        wide_n = (
            df_eff_runs[df_eff_runs["N"] == N]
            .pivot_table(
                index="run", columns="Agent", values="Efficiency (%)", aggfunc="first"
            )
            .dropna(subset=["ZI-C", "ZI-U"])
        )
        if len(wide_n) == 0:
            raw_stats.append({"N": N, "gap": np.nan, "t": np.nan, "p_raw": np.nan})
            continue
        gaps_n = wide_n["ZI-C"].to_numpy() - wide_n["ZI-U"].to_numpy()
        gap_n = gaps_n.mean()
        if stats is not None and len(wide_n) > 1:
            (t_n, p_two_n) = stats.ttest_rel(
                wide_n["ZI-C"], wide_n["ZI-U"], nan_policy="omit"
            )
            p_one_n = p_two_n / 2 if t_n > 0 else 1.0 - p_two_n / 2
        else:
            (t_n, p_one_n) = (np.nan, np.nan)
        raw_stats.append({"N": N, "gap": gap_n, "t": t_n, "p_raw": p_one_n})
    holm_p = holm_adjust([r["p_raw"] for r in raw_stats])
    all_n_significant = True
    all_n_positive_gap = True
    for i, row in enumerate(raw_stats):
        sig = np.isfinite(holm_p[i]) and holm_p[i] < 0.05
        pos = np.isfinite(row["gap"]) and row["gap"] > 0
        all_n_significant = all_n_significant and sig
        all_n_positive_gap = all_n_positive_gap and pos
        print(
            f"    N={row['N']:4d}: gap={row['gap']:.2f} pp, p_raw={row['p_raw']:.4f}, p_Holm={holm_p[i]:.4f}  {('✓ sig' if sig else '✗ not sig')}"
        )
    print("\nSensitivity Check: Efficiency Normalisation")
    print(
        "  Primary metric  : realised / (max_surplus_per_tick × n_ticks), clipped to [0, 100]"
    )
    print("  Alternative     : realised / (max_surplus_per_tick × ticks_with_trades)")
    print()
    _wide_cond = df_eff_runs.pivot_table(
        index=["N", "run"],
        columns="Agent",
        values="Cond. Efficiency (%)",
        aggfunc="first",
    ).dropna(subset=["ZI-C", "ZI-U"])
    for agent_label in ["ZI-C", "ZI-U"]:
        prim = df_results[df_results["Agent"] == agent_label]["Efficiency (%)"].mean()
        cond = (
            df_results[df_results["Agent"] == agent_label][
                "Cond. Efficiency (%)"
            ].mean()
            if "Cond. Efficiency (%)" in df_results.columns
            else np.nan
        )
        print(f"  {agent_label}:  primary={prim:.1f}%   conditional={cond:.1f}%")
    _wide_cond = _wide_cond.groupby(level="run").mean()
    if len(_wide_cond) > 1 and stats is not None:
        gaps_cond = _wide_cond["ZI-C"].to_numpy() - _wide_cond["ZI-U"].to_numpy()
        (t_cond, p_cond_two) = stats.ttest_rel(
            _wide_cond["ZI-C"], _wide_cond["ZI-U"], nan_policy="omit"
        )
        p_cond = p_cond_two / 2 if t_cond > 0 else 1.0 - p_cond_two / 2
        print(
            f"\n  Gap under alternative metric: {gaps_cond.mean():.2f} pp  (p={p_cond:.4f})"
        )
        print(
            f"  Positive significant alternative gap: {gaps_cond.mean() > 0 and p_cond < 0.05}"
        )
    print("\nConclusion")
    if all_n_positive_gap and all_n_significant:
        print(
            "  → H1 SUPPORTED: ZI-C > ZI-U at every N, Holm-corrected one-sided p < 0.05."
        )
    elif all_n_positive_gap:
        print(
            "  → H1 PARTIALLY SUPPORTED: ZI-C > ZI-U at every N, but Holm-corrected significance not met at every N."
        )
    else:
        print("  → H1 NOT SUPPORTED: ZI-C does not outperform ZI-U at all N values.")
    return (df_results, df_eff_runs)
