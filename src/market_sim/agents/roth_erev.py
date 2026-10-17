"""Extracted from the team simulator. Original contributor: Ollie James."""

from ..types import BaseAgent, Order, TradeOutcome, HistoryRecord, MarketState
import numpy as np
from typing import Optional


class RothErevAgent(BaseAgent):
    """Roth-Erev reinforcement-learning agent."""

    def __init__(
        self,
        agent_id: int,
        is_buyer: bool,
        valuation: float,
        min_price: float = 0.0,
        max_price: float = 200.0,
        n_actions: int = 21,
        recency: float = 0.1,
        experimentation: float = 0.05,
        initial_propensity: float = 1.0,
        no_trade_reward: float = 0.01,
    ):
        """Initialize agent state and configuration."""
        super().__init__(agent_id, is_buyer, valuation, min_price, max_price)
        self.n_actions = n_actions
        self.recency = recency
        self.experimentation = experimentation
        self.initial_propensity = initial_propensity
        self.no_trade_reward = no_trade_reward
        self.action_grid = np.linspace(0.0, 1.0, n_actions)
        self.propensities = np.full(n_actions, initial_propensity, dtype=float)
        self._last_action_idx: Optional[int] = None
        self._last_order_price: float = 0.0

    def get_order(self, market_state: MarketState) -> Order:
        """Convert propensities into probabilities, sample an action,"""
        probabilities = self._get_action_probabilities()
        action_idx = int(np.random.choice(self.n_actions, p=probabilities))
        aggressiveness = self.action_grid[action_idx]
        order_price = self._action_to_price(aggressiveness)
        self._last_action_idx = action_idx
        self._last_order_price = order_price
        return Order(
            agent_id=self.agent_id,
            order_type="bid" if self.is_buyer else "ask",
            price=order_price,
            quantity=1,
        )

    def observe_outcome(self, outcome: TradeOutcome) -> None:
        """Receive feedback from the market engine after each tick."""
        if outcome.accepted and outcome.profit is not None:
            self.profit += outcome.profit
            self.trade_count += 1
        reward = self._compute_reward(outcome)
        if self._last_action_idx is not None:
            self._update_propensities(self._last_action_idx, reward)

    def _get_action_probabilities(self) -> np.ndarray:
        """Convert propensities Q(a) into a valid probability distribution P(a)."""
        total = self.propensities.sum()
        if total <= 0:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        return self.propensities / total

    def _action_to_price(self, aggressiveness: float) -> float:
        """Map an aggressiveness level in [0, 1] to a feasible quote price."""
        if self.is_buyer:
            price = self.min_price + aggressiveness * (self.valuation - self.min_price)
            return float(np.clip(price, self.min_price, self.valuation))
        else:
            price = self.max_price - aggressiveness * (self.max_price - self.valuation)
            return float(np.clip(price, self.valuation, self.max_price))

    def _compute_reward(self, outcome: TradeOutcome) -> float:
        """Compute the reinforcement reward for the chosen action."""
        if outcome.accepted and outcome.profit is not None:
            return max(0.0, outcome.profit)
        return self.no_trade_reward

    def _update_propensities(self, chosen_idx: int, reward: float) -> None:
        """Update propensities using a Roth-Erev style rule."""
        self.propensities *= 1.0 - self.recency
        if self.n_actions == 1:
            self.propensities[0] += reward
            return
        spill = self.experimentation * reward / (self.n_actions - 1)
        self.propensities += spill
        self.propensities[chosen_idx] += (1.0 - self.experimentation) * reward
        self.propensities = np.maximum(self.propensities, 1e-12)

    def get_action_distribution(self) -> tuple:
        """Returns (action_grid, probability_array) for plotting."""
        return (self.action_grid.copy(), self._get_action_probabilities())

    def get_stats(self) -> dict:
        """Return agent accounting statistics."""
        probs = self._get_action_probabilities()
        best_idx = int(np.argmax(probs))
        return {
            "agent_id": self.agent_id,
            "agent_type": "RothErev",
            "is_buyer": self.is_buyer,
            "valuation": self.valuation,
            "profit": self.profit,
            "trade_count": self.trade_count,
            "avg_profit_per_trade": (
                self.profit / self.trade_count if self.trade_count > 0 else 0.0
            ),
            "best_action_idx": int(best_idx),
            "best_action": float(self.action_grid[best_idx]),
            "max_action_probability": float(probs[best_idx]),
        }
