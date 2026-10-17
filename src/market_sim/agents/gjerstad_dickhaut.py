"""Extracted from the team simulator. Original contributor: Maxime Nunes."""

from ..types import BaseAgent, Order, TradeOutcome, HistoryRecord, MarketState
import numpy as np
from typing import List


class GDAgent(BaseAgent):
    """Gjerstad-Dickhaut belief-based agent."""

    def __init__(
        self,
        agent_id: int,
        is_buyer: bool,
        valuation: float,
        min_price: float = 0.0,
        max_price: float = 200.0,
        price_steps: int = 100,
        memory_length: int = 100,
        pseudo_count: float = 1.0,
    ):
        """Configure the price grid and finite history window."""
        super().__init__(agent_id, is_buyer, valuation, min_price, max_price)
        self.price_steps = price_steps
        self.memory_length = memory_length
        self.pseudo_count = pseudo_count
        self.price_grid = np.linspace(min_price, max_price, price_steps)
        self.history: List[HistoryRecord] = []
        self._last_order_price: float = 0.0

    def get_order(self, market_state: MarketState) -> Order:
        """Quote the feasible price with maximum expected surplus."""
        belief = self._compute_belief()
        optimal_price = self._maximise_expected_surplus(belief)
        self._last_order_price = optimal_price
        return Order(
            agent_id=self.agent_id,
            order_type="bid" if self.is_buyer else "ask",
            price=optimal_price,
            quantity=1,
        )

    def observe_outcome(self, outcome: TradeOutcome) -> None:
        """Receive feedback from the market engine after each tick."""
        if outcome.accepted and outcome.profit is not None:
            self.profit += outcome.profit
            self.trade_count += 1
        if not outcome.accepted:
            self._append_history(
                HistoryRecord(
                    price=outcome.submitted_price,
                    order_type=outcome.order_type,
                    accepted=False,
                )
            )

    def observe_market_history(self, records: List[HistoryRecord]) -> None:
        """Observe accepted public orders once per tick."""
        for r in records:
            if r.accepted:
                self._append_history(r)

    def _compute_belief(self) -> np.ndarray:
        """Estimate acceptance probabilities on the price grid."""
        if not self.history:
            return np.full(self.price_steps, 0.5)
        prices = np.asarray([r.price for r in self.history])
        indices = np.abs(self.price_grid[:, None] - prices).argmin(axis=0)
        bids = np.asarray([r.order_type == "bid" for r in self.history])
        accepted = np.asarray([r.accepted for r in self.history])
        counts = lambda mask: np.bincount(indices[mask], minlength=self.price_steps)
        suffix = lambda values: np.cumsum(values[::-1])[::-1]
        alpha = suffix(counts(bids & accepted)) + np.cumsum(counts(~bids & accepted))
        beta = suffix(counts(bids & ~accepted)) + np.cumsum(counts(~bids & ~accepted))
        return (alpha + self.pseudo_count) / (alpha + beta + 2 * self.pseudo_count)

    def _maximise_expected_surplus(self, belief: np.ndarray) -> float:
        """Find the price on the grid that maximises expected surplus."""
        if self.is_buyer:
            feasible = self.price_grid <= self.valuation
            if not feasible.any():
                return self.min_price
            es = np.where(
                feasible, (self.valuation - self.price_grid) * belief, -np.inf
            )
        else:
            feasible = self.price_grid >= self.valuation
            if not feasible.any():
                return self.max_price
            es = np.where(
                feasible, (self.price_grid - self.valuation) * belief, -np.inf
            )
        best_idx = int(np.argmax(es))
        return float(self.price_grid[best_idx])

    def _append_history(self, record: HistoryRecord) -> None:
        """Add a record and trim to the sliding window."""
        self.history.append(record)
        if len(self.history) > self.memory_length:
            self.history.pop(0)

    def get_belief_curve(self) -> tuple:
        """Returns (price_grid, belief_array) for plotting."""
        return (self.price_grid.copy(), self._compute_belief())

    def get_stats(self) -> dict:
        """Return agent accounting and history length."""
        return {
            "agent_id": self.agent_id,
            "agent_type": "GD",
            "is_buyer": self.is_buyer,
            "valuation": self.valuation,
            "profit": self.profit,
            "trade_count": self.trade_count,
            "avg_profit_per_trade": (
                self.profit / self.trade_count if self.trade_count > 0 else 0.0
            ),
            "history_length": len(self.history),
        }
