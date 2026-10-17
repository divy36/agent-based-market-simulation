"""Extracted from the team simulator. Original contributor: Saul Thurley."""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict


class SimulationPipeline:
    """Post-simulation data extraction, metric computation, and CSV export."""

    def __init__(self, market, experiment_id: str = "run") -> None:
        self.market = market
        self.experiment_id = experiment_id
        self._trade_df: Optional[pd.DataFrame] = None
        self._tick_df: Optional[pd.DataFrame] = None
        self._agent_df: Optional[pd.DataFrame] = None

    @property
    def trades(self) -> pd.DataFrame:
        """Per-trade DataFrame."""
        if self._trade_df is None:
            self._trade_df = self.market.get_trade_log()
        return self._trade_df

    @property
    def ticks(self) -> pd.DataFrame:
        """Per-tick market statistics DataFrame."""
        if self._tick_df is None:
            self._tick_df = self.market.get_tick_log()
        return self._tick_df

    @property
    def agents(self) -> pd.DataFrame:
        """Per-agent final statistics DataFrame."""
        if self._agent_df is None:
            self._agent_df = self.market.get_agent_stats()
        return self._agent_df

    def rolling_volatility(self, window: int = 20) -> pd.Series:
        """Rolling standard deviation of mid-price over `window` ticks."""
        if self.ticks.empty or "mid_price" not in self.ticks.columns:
            return pd.Series(dtype=float, name="rolling_volatility")
        return (
            self.ticks.set_index("tick")["mid_price"]
            .rolling(window, min_periods=2)
            .std()
            .rename("rolling_volatility")
        )

    def rolling_avg_trade_price(self, window: int = 20) -> pd.Series:
        """Rolling mean of trade price (smoothed transaction price series)."""
        if self.trades.empty:
            return pd.Series(dtype=float, name="rolling_avg_price")
        return (
            self.trades["trade_price"]
            .rolling(window, min_periods=1)
            .mean()
            .rename("rolling_avg_price")
        )

    def price_convergence_series(self, window: int = 10) -> pd.DataFrame:
        """Return raw and smoothed trade prices."""
        if self.trades.empty:
            return pd.DataFrame(
                columns=["trade_number", "tick", "trade_price", "smoothed_price"]
            )
        df = self.trades[["tick", "trade_price"]].copy()
        df.index.name = "trade_number"
        df = df.reset_index()
        df["smoothed_price"] = df["trade_price"].rolling(window, min_periods=1).mean()
        return df

    def summary_stats(self) -> pd.DataFrame:
        """Returns a one-row DataFrame of key market-level summary statistics."""
        n_ticks = self.market.tick
        n_agents = len(self.market.all_agents)
        n_buyers = len(self.market.buyers)
        n_sellers = len(self.market.sellers)
        n_trades = len(self.trades)
        mean_price = (
            self.trades["trade_price"].mean() if not self.trades.empty else np.nan
        )
        std_price = (
            self.trades["trade_price"].std() if not self.trades.empty else np.nan
        )
        spread_series = (
            self.ticks["spread"].dropna() if not self.ticks.empty else pd.Series()
        )
        mean_spread = spread_series.mean() if not spread_series.empty else np.nan
        std_spread = spread_series.std() if not spread_series.empty else np.nan
        mid_series = (
            self.ticks["mid_price"].dropna() if not self.ticks.empty else pd.Series()
        )
        mid_vol = mid_series.std() if not mid_series.empty else np.nan
        if not self.agents.empty and "is_buyer" in self.agents.columns:
            buyer_profits = self.agents.loc[self.agents["is_buyer"] == True, "profit"]
            seller_profits = self.agents.loc[self.agents["is_buyer"] == False, "profit"]
        else:
            buyer_profits = pd.Series(dtype=float)
            seller_profits = pd.Series(dtype=float)
        return pd.DataFrame(
            [
                {
                    "experiment_id": self.experiment_id,
                    "n_agents": n_agents,
                    "n_buyers": n_buyers,
                    "n_sellers": n_sellers,
                    "n_ticks": n_ticks,
                    "total_trades": n_trades,
                    "trade_rate": n_trades / n_ticks if n_ticks > 0 else 0.0,
                    "mean_trade_price": (
                        round(mean_price, 4) if not np.isnan(mean_price) else np.nan
                    ),
                    "std_trade_price": (
                        round(std_price, 4) if not np.isnan(std_price) else np.nan
                    ),
                    "mean_spread": (
                        round(mean_spread, 4) if not np.isnan(mean_spread) else np.nan
                    ),
                    "std_spread": (
                        round(std_spread, 4) if not np.isnan(std_spread) else np.nan
                    ),
                    "mid_price_volatility": (
                        round(mid_vol, 4) if not np.isnan(mid_vol) else np.nan
                    ),
                    "allocative_efficiency": round(self.allocative_efficiency(), 2),
                    "mean_buyer_profit": (
                        round(buyer_profits.mean(), 4)
                        if not buyer_profits.empty
                        else np.nan
                    ),
                    "mean_seller_profit": (
                        round(seller_profits.mean(), 4)
                        if not seller_profits.empty
                        else np.nan
                    ),
                }
            ]
        )

    def allocative_efficiency(self) -> float:
        """Allocative efficiency as a percentage."""
        buyer_vals = sorted([b.valuation for b in self.market.buyers], reverse=True)
        seller_costs = sorted([s.valuation for s in self.market.sellers], reverse=False)
        max_surplus = 0.0
        for v, c in zip(buyer_vals, seller_costs):
            if v >= c:
                max_surplus += v - c
            else:
                break
        if max_surplus <= 0.0 or self.market.tick <= 0:
            return 0.0
        realised = sum((a.profit for a in self.market.all_agents))
        return max(0.0, min(100.0, realised / (max_surplus * self.market.tick) * 100.0))

    def profits_by_agent_type(self) -> pd.DataFrame:
        """Mean, total, and trade-count grouped by agent type."""
        if self.agents.empty or "agent_type" not in self.agents.columns:
            return pd.DataFrame()
        grouped = self.agents.groupby("agent_type").agg(
            n_agents=("agent_id", "count"),
            mean_profit=("profit", "mean"),
            total_profit=("profit", "sum"),
            mean_trade_count=("trade_count", "mean"),
        )
        if "avg_profit_per_trade" in self.agents.columns:
            apt = self.agents.groupby("agent_type")["avg_profit_per_trade"].mean()
            grouped["mean_avg_profit_per_trade"] = apt
        return grouped.reset_index()

    def spread_over_time(self) -> pd.Series:
        """Returns the per-tick spread series (best_ask - best_bid)."""
        if self.ticks.empty or "spread" not in self.ticks.columns:
            return pd.Series(dtype=float, name="spread")
        return self.ticks.set_index("tick")["spread"].rename("spread")

    def export_all(self, output_dir: str = ".") -> None:
        """Export all simulation data to CSV files in output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, self.experiment_id)
        files_written = []
        self.trades.to_csv(f"{prefix}_trades.csv", index=False)
        files_written.append(f"{self.experiment_id}_trades.csv")
        self.ticks.to_csv(f"{prefix}_ticks.csv", index=False)
        files_written.append(f"{self.experiment_id}_ticks.csv")
        self.agents.to_csv(f"{prefix}_agents.csv", index=False)
        files_written.append(f"{self.experiment_id}_agents.csv")
        self.summary_stats().to_csv(f"{prefix}_summary.csv", index=False)
        files_written.append(f"{self.experiment_id}_summary.csv")
        vol_df = self.rolling_volatility().reset_index()
        vol_df.to_csv(f"{prefix}_volatility.csv", index=False)
        files_written.append(f"{self.experiment_id}_volatility.csv")
        print(f"[DataPipeline] Exported to '{output_dir}/':")
        for f in files_written:
            print(f"    {f}")

    def export_trades(self, path: str) -> None:
        """Export just the trade log to a specific path."""
        self.trades.to_csv(path, index=False)
        print(f"[DataPipeline] Trade log → {path}  ({len(self.trades)} rows)")

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return all DataFrames as a dict for programmatic access."""
        return {
            "trades": self.trades,
            "ticks": self.ticks,
            "agents": self.agents,
            "summary": self.summary_stats(),
            "volatility": self.rolling_volatility().reset_index(),
        }

    def __repr__(self) -> str:
        return f"SimulationPipeline(experiment='{self.experiment_id}', trades={len(self.trades)}, ticks={len(self.ticks)})"


def compare_experiments(experiments: Dict) -> pd.DataFrame:
    """Run a cross-experiment comparison."""
    rows = []
    for exp_id, market in experiments.items():
        pipe = SimulationPipeline(market, experiment_id=exp_id)
        rows.append(pipe.summary_stats())
    return pd.concat(rows, ignore_index=True)
