"""Extracted from the team simulator. Original contributor: Team shared interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    """A single bid or ask submitted to the order book."""

    agent_id: int
    order_type: str
    price: float
    quantity: int = 1


@dataclass
class TradeOutcome:
    """Feedback sent back to an agent after each round."""

    agent_id: int
    submitted_price: float
    order_type: str
    accepted: bool
    trade_price: Optional[float] = None
    profit: Optional[float] = None


@dataclass
class HistoryRecord:
    """One market event entry - used by GD agents to build their belief function."""

    price: float
    order_type: str
    accepted: bool


@dataclass
class MarketState:
    """Snapshot of the order book passed to agents before they quote."""

    best_bid: Optional[float]
    best_ask: Optional[float]
    last_trade_price: Optional[float]
    tick: int


class BaseAgent(ABC):
    """Every agent type inherits from this class."""

    def __init__(
        self,
        agent_id: int,
        is_buyer: bool,
        valuation: float,
        min_price: float,
        max_price: float,
    ):
        """Initialize agent state and configuration."""
        self.agent_id = agent_id
        self.is_buyer = is_buyer
        self.valuation = valuation
        self.min_price = min_price
        self.max_price = max_price
        self.profit = 0.0
        self.trade_count = 0

    @abstractmethod
    def get_order(self, market_state: MarketState) -> Order:
        """Called once per tick. Must return an Order to submit to the book."""
        ...

    @abstractmethod
    def observe_outcome(self, outcome: TradeOutcome) -> None:
        """Called once per tick with the result of this agent's last order."""
        ...

    def __repr__(self):
        role = "Buyer" if self.is_buyer else "Seller"
        return f"{self.__class__.__name__}(id={self.agent_id}, {role}, val={self.valuation:.2f})"
