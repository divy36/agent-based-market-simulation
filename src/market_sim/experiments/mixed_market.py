"""Original experiment: Ollie James; packaged with statistical corrections."""
from ..market import DoubleAuctionMarket
from ..pipeline import SimulationPipeline
from ..agents import ZICAgent, ZIUAgent, GDAgent, RothErevAgent


def run(n_ticks=400, seeds=(11, 22, 33, 44, 55)):
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    SEEDS = seeds
    N_PER_TYPE = 4
    N_HOMO = 12
    N_TICKS = n_ticks
    MIN_PRICE = 0
    MAX_PRICE = 200
    EQ_PRICE = 100

    def build_agents(agent_cls, n, is_buyer, start_id, rng, min_price=0, max_price=200):
        vals = [
            rng.uniform(100, 200) if is_buyer else rng.uniform(0, 100) for _ in range(n)
        ]
        return [
            agent_cls(start_id + i, is_buyer, vals[i], min_price, max_price)
            for i in range(n)
        ]

    def build_mixed_agents(n_per_type, rng, min_price=0, max_price=200):
        buyers = (
            build_agents(ZICAgent, n_per_type, True, 0, rng, min_price, max_price)
            + build_agents(
                RothErevAgent, n_per_type, True, 1000, rng, min_price, max_price
            )
            + build_agents(GDAgent, n_per_type, True, 2000, rng, min_price, max_price)
        )
        sellers = (
            build_agents(ZICAgent, n_per_type, False, 3000, rng, min_price, max_price)
            + build_agents(
                RothErevAgent, n_per_type, False, 4000, rng, min_price, max_price
            )
            + build_agents(GDAgent, n_per_type, False, 5000, rng, min_price, max_price)
        )
        return (buyers, sellers)

    def run_market(buyers, sellers, seed):
        random.seed(seed)
        np.random.seed(seed)
        market = DoubleAuctionMarket(
            buyers, sellers, min_price=MIN_PRICE, max_price=MAX_PRICE, random_seed=seed
        )
        market.run(N_TICKS)
        trade_df = market.get_trade_log()
        tick_df = market.get_tick_log()
        agent_df = market.get_agent_stats()
        return (market, trade_df, tick_df, agent_df)

    def market_metrics(trade_df, tick_df, agent_df, condition_name):
        out = {
            "condition": condition_name,
            "n_trades": len(trade_df),
            "avg_trade_price": np.nan,
            "price_volatility": np.nan,
            "mean_abs_deviation_from_eq": np.nan,
            "mean_spread": np.nan,
            "total_surplus": np.nan,
            "mean_profit": np.nan,
        }
        if not trade_df.empty:
            out["avg_trade_price"] = trade_df["trade_price"].mean()
            out["price_volatility"] = trade_df["trade_price"].std()
            out["mean_abs_deviation_from_eq"] = (
                (trade_df["trade_price"] - EQ_PRICE).abs().mean()
            )
        if "spread" in tick_df.columns:
            out["mean_spread"] = tick_df["spread"].dropna().mean()
        if not agent_df.empty:
            out["total_surplus"] = agent_df["profit"].sum()
            out["mean_profit"] = agent_df["profit"].mean()
        return out

    def add_agent_level_metrics(agent_df, condition_name, mixed_only=False):
        df = agent_df.copy()
        df["condition"] = condition_name
        label_map = {
            "ZICAgent": "ZI-C",
            "RothErevAgent": "Roth-Erev",
            "GDAgent": "GD",
            "ZIC": "ZI-C",
            "RothErev": "Roth-Erev",
        }
        df["agent_label"] = df["agent_type"].map(label_map).fillna(df["agent_type"])
        total_surplus = df["profit"].sum()
        if total_surplus != 0:
            df["surplus_share"] = df["profit"] / total_surplus
        else:
            df["surplus_share"] = 0.0
        if mixed_only:
            keep = ["ZI-C", "Roth-Erev", "GD"]
            df = df[df["agent_label"].isin(keep)]
        return df

    all_market_results = []
    all_agent_results = []
    example_runs = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        conditions = {
            "ZIC-only": (
                build_agents(ZICAgent, N_HOMO, True, 0, rng, MIN_PRICE, MAX_PRICE),
                build_agents(ZICAgent, N_HOMO, False, 1000, rng, MIN_PRICE, MAX_PRICE),
            ),
            "RothErev-only": (
                build_agents(
                    RothErevAgent, N_HOMO, True, 2000, rng, MIN_PRICE, MAX_PRICE
                ),
                build_agents(
                    RothErevAgent, N_HOMO, False, 3000, rng, MIN_PRICE, MAX_PRICE
                ),
            ),
            "GD-only": (
                build_agents(GDAgent, N_HOMO, True, 4000, rng, MIN_PRICE, MAX_PRICE),
                build_agents(GDAgent, N_HOMO, False, 5000, rng, MIN_PRICE, MAX_PRICE),
            ),
            "Mixed-All": build_mixed_agents(N_PER_TYPE, rng, MIN_PRICE, MAX_PRICE),
        }
        for condition_name, (buyers, sellers) in conditions.items():
            (market, trade_df, tick_df, agent_df) = run_market(buyers, sellers, seed)
            all_market_results.append(
                market_metrics(trade_df, tick_df, agent_df, condition_name)
            )
            all_market_results[-1]["seed"] = seed
            agent_res = add_agent_level_metrics(
                agent_df, condition_name, mixed_only=condition_name == "Mixed-All"
            )
            agent_res["seed"] = seed
            all_agent_results.append(agent_res)
            if seed == SEEDS[0]:
                example_runs[condition_name] = {
                    "market": market,
                    "trade_df": trade_df,
                    "tick_df": tick_df,
                    "agent_df": agent_df,
                }
    market_results_df = pd.DataFrame(all_market_results)
    agent_results_df = pd.concat(all_agent_results, ignore_index=True)
    market_summary = market_results_df.groupby("condition", as_index=False).agg(
        {
            "n_trades": ["mean", "std"],
            "avg_trade_price": ["mean", "std"],
            "price_volatility": ["mean", "std"],
            "mean_abs_deviation_from_eq": ["mean", "std"],
            "mean_spread": ["mean", "std"],
            "total_surplus": ["mean", "std"],
        }
    )
    market_summary.columns = [
        "_".join(col).strip("_") for col in market_summary.columns.to_flat_index()
    ]
    mixed_agent_summary = (
        agent_results_df[agent_results_df["condition"] == "Mixed-All"]
        .groupby("agent_label", as_index=False)
        .agg(
            {
                "profit": ["mean", "std"],
                "trade_count": ["mean", "std"],
                "surplus_share": ["mean", "std"],
            }
        )
    )
    mixed_agent_summary.columns = [
        "_".join(col).strip("_") for col in mixed_agent_summary.columns.to_flat_index()
    ]
    print("=" * 70)
    print("EXPERIMENT 2 - FULLY MIXED MARKET")
    print("=" * 70)
    print("\nRQ1: Market-level comparison")
    print(market_summary.round(3))
    print("\nRQ2: Agent-level outcomes in the fully mixed market")
    print(mixed_agent_summary.round(3))
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    for condition_name, payload in example_runs.items():
        trade_df = payload["trade_df"]
        if not trade_df.empty:
            series = trade_df["trade_price"].rolling(20).mean()
            ax1.plot(trade_df["tick"], series, lw=2, label=condition_name)
    ax1.axhline(EQ_PRICE, color="red", lw=1.2, ls="--", label="Equilibrium")
    ax1.set_title("RQ1: Trade Price Convergence")
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("20-trade Moving Average Price")
    ax1.legend(fontsize=8)
    ax2 = fig.add_subplot(gs[0, 1])
    vol_df = (
        market_results_df.groupby("condition", as_index=False)["price_volatility"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax2.bar(
        vol_df["condition"], vol_df["mean"], yerr=vol_df["std"], capsize=5, alpha=0.8
    )
    ax2.set_title("RQ1: Price Volatility by Market Type")
    ax2.set_ylabel("Std. Dev. of Trade Price")
    ax2.tick_params(axis="x", rotation=20)
    ax3 = fig.add_subplot(gs[1, 0])
    spread_df = (
        market_results_df.groupby("condition", as_index=False)["mean_spread"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax3.bar(
        spread_df["condition"],
        spread_df["mean"],
        yerr=spread_df["std"],
        capsize=5,
        alpha=0.8,
    )
    ax3.set_title("RQ1: Mean Bid-Ask Spread by Market Type")
    ax3.set_ylabel("Mean Spread")
    ax3.tick_params(axis="x", rotation=20)
    ax4 = fig.add_subplot(gs[1, 1])
    surplus_df = (
        market_results_df.groupby("condition", as_index=False)["total_surplus"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax4.bar(
        surplus_df["condition"],
        surplus_df["mean"],
        yerr=surplus_df["std"],
        capsize=5,
        alpha=0.8,
    )
    ax4.set_title("RQ1: Total Surplus by Market Type")
    ax4.set_ylabel("Total Surplus")
    ax4.tick_params(axis="x", rotation=20)
    ax5 = fig.add_subplot(gs[2, 0])
    profit_mixed = agent_results_df[agent_results_df["condition"] == "Mixed-All"]
    profit_plot = (
        profit_mixed.groupby("agent_label", as_index=False)["profit"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax5.bar(
        profit_plot["agent_label"],
        profit_plot["mean"],
        yerr=profit_plot["std"],
        capsize=5,
        alpha=0.85,
    )
    ax5.set_title("RQ2: Mean Profit by Agent Type (Mixed Market)")
    ax5.set_ylabel("Profit")
    ax6 = fig.add_subplot(gs[2, 1])
    trade_plot = (
        profit_mixed.groupby("agent_label", as_index=False)["trade_count"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax6.bar(
        trade_plot["agent_label"],
        trade_plot["mean"],
        yerr=trade_plot["std"],
        capsize=5,
        alpha=0.85,
    )
    ax6.set_title("RQ2: Mean Trade Count by Agent Type (Mixed Market)")
    ax6.set_ylabel("Trades per Agent")
    plt.suptitle(
        "Experiment 2 - Homogeneous and Mixed Markets",
        fontsize=13,
        fontweight="bold",
    )
    print("Saved: experiment2_mixed_market.png")
    surplus_capture = (
        agent_results_df[agent_results_df["condition"] == "Mixed-All"]
        .groupby(["seed", "agent_label"])["surplus_share"]
        .sum()
        .groupby("agent_label")
        .agg(["mean", "std"])
        .reset_index()
    )
    print("\nSurplus capture in Mixed-All:")
    print(surplus_capture.round(3))
    return (market_results_df, agent_results_df)
