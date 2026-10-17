"""Extracted from the team simulator. Original contributor: Saul Thurley."""

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class Trade:
    """A matched transaction recorded by the order book."""

    tick: int
    buyer_id: int
    seller_id: int
    bid_price: float
    ask_price: float
    trade_price: float
    quantity: int = 1


class OrderBook:
    """Continuous double auction order book with price-time priority."""

    def __init__(self) -> None:
        self._bids: List[Tuple] = []
        self._asks: List[Tuple] = []
        self._seq: int = 0
        self._current_tick: int = 0

    def set_tick(self, tick: int) -> None:
        """Update the tick number stamped on every Trade produced."""
        self._current_tick = tick

    def clear(self) -> None:
        """Cancel all resting orders and reset the sequence counter."""
        self._bids.clear()
        self._asks.clear()
        self._seq = 0

    @property
    def best_bid(self) -> Optional[float]:
        """Highest outstanding bid price, or None if the bid side is empty."""
        if self._bids:
            return -self._bids[0][0]
        return None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest outstanding ask price, or None if the ask side is empty."""
        if self._asks:
            return self._asks[0][0]
        return None

    @property
    def spread(self) -> Optional[float]:
        """Best ask − best bid. None if either side of the book is empty."""
        (bb, ba) = (self.best_bid, self.best_ask)
        if bb is not None and ba is not None:
            return ba - bb
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """(Best bid + best ask) / 2. None if either side is empty."""
        (bb, ba) = (self.best_bid, self.best_ask)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    def get_depth_snapshot(self) -> Dict:
        """Returns a dict summarising the current state of the order book."""
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "bid_depth": len(self._bids),
            "ask_depth": len(self._asks),
        }

    def submit_order(self, order) -> List[Trade]:
        """Submit an order to the book."""
        if order.quantity != 1 or not math.isfinite(order.price) or order.price < 0:
            raise ValueError(
                "Orders require a finite nonnegative price and quantity one"
            )
        self._seq += 1
        if order.order_type == "bid":
            return self._process_bid(order)
        elif order.order_type == "ask":
            return self._process_ask(order)
        else:
            raise ValueError(
                f"Unknown order_type: {order.order_type!r}. Expected 'bid' or 'ask'."
            )

    def _process_bid(self, bid) -> List[Trade]:
        """Try to match an incoming bid against resting asks (price-time priority)."""
        if self._asks and bid.price >= self._asks[0][0]:
            (ask_price, _ask_seq, resting_ask) = heapq.heappop(self._asks)
            trade = Trade(
                tick=self._current_tick,
                buyer_id=bid.agent_id,
                seller_id=resting_ask.agent_id,
                bid_price=bid.price,
                ask_price=ask_price,
                trade_price=ask_price,
            )
            return [trade]
        heapq.heappush(self._bids, (-bid.price, self._seq, bid))
        return []

    def _process_ask(self, ask) -> List[Trade]:
        """Try to match an incoming ask against resting bids (price-time priority)."""
        if self._bids and ask.price <= -self._bids[0][0]:
            (neg_bid_price, _bid_seq, resting_bid) = heapq.heappop(self._bids)
            bid_price = -neg_bid_price
            trade = Trade(
                tick=self._current_tick,
                buyer_id=resting_bid.agent_id,
                seller_id=ask.agent_id,
                bid_price=bid_price,
                ask_price=ask.price,
                trade_price=bid_price,
            )
            return [trade]
        heapq.heappush(self._asks, (ask.price, self._seq, ask))
        return []
