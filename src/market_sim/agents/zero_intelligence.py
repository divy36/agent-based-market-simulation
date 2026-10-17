"""Extracted from the team simulator. Original contributor: Divy Chauhan."""

from ..types import BaseAgent, Order, TradeOutcome, HistoryRecord, MarketState
import random


def _rounded_price_with_bounds(price: float, low: float, high: float) -> float:
    """Round to 2dp while preserving the closed interval [low, high]."""
    if low > high:
        (low, high) = (high, low)
    p = round(price, 2)
    return min(max(p, low), high)


def _validate_price_bounds(min_price: float, max_price: float) -> None:
    """Basic guardrails so random draws always have a valid interval."""
    if min_price > max_price:
        raise ValueError(
            f"min_price ({min_price}) cannot exceed max_price ({max_price})."
        )


class ZICAgent(BaseAgent):
    """Zero Intelligence Constrained agent - Gode & Sunder (1993)."""

    def __init__(
        self,
        agent_id: int,
        is_buyer: bool,
        valuation: float,
        min_price: float,
        max_price: float,
    ):
        super().__init__(agent_id, is_buyer, valuation, min_price, max_price)
        _validate_price_bounds(self.min_price, self.max_price)
        if self.is_buyer and self.valuation < self.min_price:
            raise ValueError(
                f"ZI-C buyer valuation ({self.valuation}) must be >= min_price ({self.min_price})."
            )
        if not self.is_buyer and self.valuation > self.max_price:
            raise ValueError(
                f"ZI-C seller cost ({self.valuation}) must be <= max_price ({self.max_price})."
            )

    def get_order(self, market_state: MarketState) -> Order:
        if self.is_buyer:
            (low, high) = (self.min_price, self.valuation)
        else:
            (low, high) = (self.valuation, self.max_price)
        price = random.uniform(low, high)
        price = _rounded_price_with_bounds(price, low, high)
        return Order(self.agent_id, "bid" if self.is_buyer else "ask", price)

    def observe_outcome(self, outcome: TradeOutcome) -> None:
        if outcome.accepted and outcome.profit is not None:
            self.profit += outcome.profit
            self.trade_count += 1

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": "ZIC",
            "is_buyer": self.is_buyer,
            "valuation": self.valuation,
            "profit": self.profit,
            "trade_count": self.trade_count,
            "avg_profit_per_trade": (
                self.profit / self.trade_count if self.trade_count > 0 else 0.0
            ),
        }


class ZIUAgent(BaseAgent):
    """Zero Intelligence Unconstrained agent - Gode & Sunder (1993)."""

    def __init__(
        self,
        agent_id: int,
        is_buyer: bool,
        valuation: float,
        min_price: float,
        max_price: float,
    ):
        super().__init__(agent_id, is_buyer, valuation, min_price, max_price)
        _validate_price_bounds(self.min_price, self.max_price)

    def get_order(self, market_state: MarketState) -> Order:
        (low, high) = (self.min_price, self.max_price)
        price = random.uniform(low, high)
        price = _rounded_price_with_bounds(price, low, high)
        return Order(self.agent_id, "bid" if self.is_buyer else "ask", price)

    def observe_outcome(self, outcome: TradeOutcome) -> None:
        if outcome.accepted and outcome.profit is not None:
            self.profit += outcome.profit
            self.trade_count += 1

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": "ZIU",
            "is_buyer": self.is_buyer,
            "valuation": self.valuation,
            "profit": self.profit,
            "trade_count": self.trade_count,
            "avg_profit_per_trade": (
                self.profit / self.trade_count if self.trade_count > 0 else 0.0
            ),
        }
