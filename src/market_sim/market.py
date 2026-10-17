"""Extracted from the team simulator. Original contributor: Saul Thurley."""

from .order_book import OrderBook, Trade
from .types import BaseAgent, Order, TradeOutcome, HistoryRecord, MarketState
import random
import numpy as np
from typing import List, Dict, Optional


class DoubleAuctionMarket:
    """Full continuous double auction simulation engine."""

    def __init__(
        self,
        buyers: List,
        sellers: List,
        min_price: float = 0.0,
        max_price: float = 200.0,
        clear_book_each_tick: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        if not clear_book_each_tick:
            raise ValueError("Only per-tick order clearing is supported")
        agents = buyers + sellers
        if len({a.agent_id for a in agents}) != len(agents):
            raise ValueError("Agent IDs must be unique")
        if any(not a.is_buyer for a in buyers) or any(a.is_buyer for a in sellers):
            raise ValueError("Agent side does not match its market population")
        self.buyers = buyers
        self.sellers = sellers
        self.all_agents = buyers + sellers
        self.min_price = min_price
        self.max_price = max_price
        self.clear_book_each_tick = clear_book_each_tick
        self.order_book = OrderBook()
        self.tick = 0
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self._last_trade_price: Optional[float] = None
        self.trade_log: List[Dict] = []
        self.tick_log: List[Dict] = []

    def run(self, n_ticks: int) -> None:
        """Execute the simulation for exactly n_ticks periods."""
        if not isinstance(n_ticks, int) or n_ticks < 0:
            raise ValueError("n_ticks must be a nonnegative integer")
        for _ in range(n_ticks):
            self._run_tick()

    def _run_tick(self) -> None:
        """Execute one simulation period."""
        if self.clear_book_each_tick:
            self.order_book.clear()
        self.order_book.set_tick(self.tick)
        state = MarketState(
            best_bid=self.order_book.best_bid,
            best_ask=self.order_book.best_ask,
            last_trade_price=self._last_trade_price,
            tick=self.tick,
        )
        agent_orders: List = [
            (agent, agent.get_order(state)) for agent in self.all_agents
        ]
        random.shuffle(agent_orders)
        order_registry: Dict[int, object] = {
            agent.agent_id: order for (agent, order) in agent_orders
        }
        agent_registry: Dict[int, object] = {
            agent.agent_id: agent for (agent, _) in agent_orders
        }
        tick_trades: List = []
        for _agent, order in agent_orders:
            trades = self.order_book.submit_order(order)
            if trades:
                tick_trades.extend(trades)
                self._last_trade_price = trades[-1].trade_price
        traded_ids = {t.buyer_id for t in tick_trades} | {
            t.seller_id for t in tick_trades
        }
        self._dispatch_outcomes(tick_trades, traded_ids, agent_registry, order_registry)
        tick_history = self._build_history(agent_orders, traded_ids)
        for agent in self.all_agents:
            if hasattr(agent, "observe_market_history"):
                agent.observe_market_history(tick_history)
        self._log_tick(tick_trades)
        self.tick += 1

    def _dispatch_outcomes(
        self,
        tick_trades: List,
        traded_ids: set,
        agent_registry: Dict,
        order_registry: Dict,
    ) -> None:
        """Send a TradeOutcome to every agent after the tick resolves."""
        for trade in tick_trades:
            buyer = agent_registry.get(trade.buyer_id)
            if buyer is not None:
                buyer_order = order_registry.get(trade.buyer_id)
                submitted = buyer_order.price if buyer_order else trade.bid_price
                profit = buyer.valuation - trade.trade_price
                buyer.observe_outcome(
                    TradeOutcome(
                        agent_id=trade.buyer_id,
                        submitted_price=submitted,
                        order_type="bid",
                        accepted=True,
                        trade_price=trade.trade_price,
                        profit=profit,
                    )
                )
            seller = agent_registry.get(trade.seller_id)
            if seller is not None:
                seller_order = order_registry.get(trade.seller_id)
                submitted = seller_order.price if seller_order else trade.ask_price
                profit = trade.trade_price - seller.valuation
                seller.observe_outcome(
                    TradeOutcome(
                        agent_id=trade.seller_id,
                        submitted_price=submitted,
                        order_type="ask",
                        accepted=True,
                        trade_price=trade.trade_price,
                        profit=profit,
                    )
                )
        for agent_id, agent in agent_registry.items():
            if agent_id not in traded_ids:
                order = order_registry[agent_id]
                agent.observe_outcome(
                    TradeOutcome(
                        agent_id=agent_id,
                        submitted_price=order.price,
                        order_type=order.order_type,
                        accepted=False,
                        trade_price=None,
                        profit=None,
                    )
                )

    def _build_history(self, agent_orders: List, traded_ids: set) -> List:
        """Build a list of HistoryRecord objects for this tick."""
        history = []
        for agent, order in agent_orders:
            history.append(
                HistoryRecord(
                    price=order.price,
                    order_type=order.order_type,
                    accepted=agent.agent_id in traded_ids,
                )
            )
        return history

    def _log_tick(self, tick_trades: List) -> None:
        """Append one row to trade_log (per trade) and one row to tick_log."""
        for trade in tick_trades:
            self.trade_log.append(
                {
                    "tick": self.tick,
                    "buyer_id": trade.buyer_id,
                    "seller_id": trade.seller_id,
                    "bid_price": trade.bid_price,
                    "ask_price": trade.ask_price,
                    "trade_price": trade.trade_price,
                }
            )
        depth = self.order_book.get_depth_snapshot()
        n_trades = len(tick_trades)
        avg_price = (
            float(np.mean([t.trade_price for t in tick_trades]))
            if tick_trades
            else None
        )
        self.tick_log.append(
            {
                "tick": self.tick,
                "n_trades": n_trades,
                "avg_trade_price": avg_price,
                "best_bid": depth["best_bid"],
                "best_ask": depth["best_ask"],
                "spread": depth["spread"],
                "mid_price": depth["mid_price"],
                "last_trade_price": self._last_trade_price,
            }
        )

    def get_trade_log(self):
        """Returns a pandas DataFrame with one row per executed trade."""
        import pandas as pd

        return pd.DataFrame(self.trade_log)

    def get_tick_log(self):
        """Returns a pandas DataFrame with one row per tick."""
        import pandas as pd

        return pd.DataFrame(self.tick_log)

    def get_agent_stats(self):
        """Returns a pandas DataFrame with one row per agent (final stats)."""
        import pandas as pd

        rows = []
        for agent in self.all_agents:
            if hasattr(agent, "get_stats"):
                row = agent.get_stats()
            else:
                count = agent.trade_count
                rows.append(
                    {
                        "agent_id": agent.agent_id,
                        "agent_type": agent.__class__.__name__,
                        "is_buyer": agent.is_buyer,
                        "valuation": agent.valuation,
                        "profit": agent.profit,
                        "trade_count": count,
                        "avg_profit_per_trade": (
                            agent.profit / count if count > 0 else 0.0
                        ),
                    }
                )
                continue
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"DoubleAuctionMarket(buyers={len(self.buyers)}, sellers={len(self.sellers)}, tick={self.tick}, trades={len(self.trade_log)})"
