from market_sim.types import *
from market_sim.agents import *
from market_sim.market import DoubleAuctionMarket
import random


def test_basic():
    zic_buyer = ZICAgent(
        agent_id=0, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    zic_seller = ZICAgent(
        agent_id=1, is_buyer=False, valuation=50.0, min_price=1, max_price=200
    )
    ziu_buyer = ZIUAgent(
        agent_id=2, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    ziu_seller = ZIUAgent(
        agent_id=3, is_buyer=False, valuation=50.0, min_price=1, max_price=200
    )
    state = MarketState(best_bid=None, best_ask=None, last_trade_price=None, tick=0)
    for agent, expected_type, label in [
        (zic_buyer, "bid", "ZI-C buyer"),
        (zic_seller, "ask", "ZI-C seller"),
        (ziu_buyer, "bid", "ZI-U buyer"),
        (ziu_seller, "ask", "ZI-U seller"),
    ]:
        order = agent.get_order(state)
        assert order.agent_id == agent.agent_id, f"{label}: wrong agent_id"
        assert order.order_type == expected_type, f"{label}: wrong order_type"
        assert order.price >= agent.min_price, f"{label}: price below min_price"
        assert order.price <= agent.max_price, f"{label}: price above max_price"
    try:
        ZICAgent(agent_id=99, is_buyer=True, valuation=0.5, min_price=1, max_price=200)
        assert False, "Expected invalid ZI-C buyer config to raise ValueError"
    except ValueError:
        pass
    try:
        ZIUAgent(agent_id=98, is_buyer=True, valuation=100, min_price=200, max_price=1)
        assert False, "Expected invalid ZIP bounds to raise ValueError"
    except ValueError:
        pass


def test_zic_constraint():
    buyer = ZICAgent(
        agent_id=0, is_buyer=True, valuation=99.97, min_price=1, max_price=200
    )
    seller = ZICAgent(
        agent_id=1, is_buyer=False, valuation=50.03, min_price=1, max_price=200
    )
    state = MarketState(best_bid=None, best_ask=None, last_trade_price=None, tick=0)
    for _ in range(500):
        b = buyer.get_order(state)
        s = seller.get_order(state)
        assert (
            b.price <= buyer.valuation
        ), f"ZI-C buyer bid {b.price} > valuation {buyer.valuation}"
        assert (
            s.price >= seller.valuation
        ), f"ZI-C seller ask {s.price} < cost {seller.valuation}"


def test_ziu_unconstrained():
    rng_state = random.getstate()
    random.seed(42)
    buyer = ZIUAgent(
        agent_id=0, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    seller = ZIUAgent(
        agent_id=1, is_buyer=False, valuation=50.0, min_price=1, max_price=200
    )
    state = MarketState(best_bid=None, best_ask=None, last_trade_price=None, tick=0)
    buyer_prices = [buyer.get_order(state).price for _ in range(500)]
    seller_prices = [seller.get_order(state).price for _ in range(500)]
    random.setstate(rng_state)
    assert (
        max(buyer_prices) > 150
    ), "ZI-U buyer never exceeded valuation - should be unconstrained"
    assert (
        min(seller_prices) < 50
    ), "ZI-U seller never went below cost - should be unconstrained"


def test_profit_accumulation():
    zic_buyer = ZICAgent(
        agent_id=0, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    zic_seller = ZICAgent(
        agent_id=1, is_buyer=False, valuation=50.0, min_price=1, max_price=200
    )
    ziu_buyer = ZIUAgent(
        agent_id=2, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    ziu_seller = ZIUAgent(
        agent_id=3, is_buyer=False, valuation=50.0, min_price=1, max_price=200
    )
    for buyer, seller in [(zic_buyer, zic_seller), (ziu_buyer, ziu_seller)]:
        buyer.observe_outcome(
            TradeOutcome(buyer.agent_id, 110.0, "bid", True, 100.0, 50.0)
        )
        seller.observe_outcome(
            TradeOutcome(seller.agent_id, 90.0, "ask", True, 100.0, 50.0)
        )
        assert buyer.profit == 50.0, f"Expected 50.0, got {buyer.profit}"
        assert seller.profit == 50.0, f"Expected 50.0, got {seller.profit}"
        assert buyer.trade_count == 1
        assert seller.trade_count == 1
    multi = ZICAgent(
        agent_id=4, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    multi.observe_outcome(TradeOutcome(4, 110.0, "bid", True, 95.0, 55.0))
    multi.observe_outcome(TradeOutcome(4, 110.0, "bid", True, 105.0, 45.0))
    assert (
        multi.profit == 100.0
    ), f"Multi-trade profit wrong: expected 100.0, got {multi.profit}"
    assert multi.trade_count == 2


def test_rejected_order():
    zic = ZICAgent(
        agent_id=0, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    ziu_ = ZIUAgent(
        agent_id=1, is_buyer=True, valuation=150.0, min_price=1, max_price=200
    )
    for agent in [zic, ziu_]:
        agent.observe_outcome(TradeOutcome(agent.agent_id, 80.0, "bid", accepted=False))
        assert agent.profit == 0.0, f"Profit changed on rejected order: {agent.profit}"
        assert (
            agent.trade_count == 0
        ), f"Trade count changed on rejected order: {agent.trade_count}"


def test_boundary_edges():
    state = MarketState(best_bid=None, best_ask=None, last_trade_price=None, tick=0)
    buyer_at_min = ZICAgent(
        agent_id=0, is_buyer=True, valuation=1.0, min_price=1, max_price=200
    )
    for _ in range(20):
        o = buyer_at_min.get_order(state)
        assert o.price == 1.0, f"Buyer at min_price: expected bid=1.0, got {o.price}"
        assert o.price >= buyer_at_min.min_price, "Bid below min_price"
        assert o.price <= buyer_at_min.valuation, "Bid above valuation"
    seller_at_max = ZICAgent(
        agent_id=1, is_buyer=False, valuation=200.0, min_price=1, max_price=200
    )
    for _ in range(20):
        o = seller_at_max.get_order(state)
        assert (
            o.price == 200.0
        ), f"Seller at max_price: expected ask=200.0, got {o.price}"
        assert o.price >= seller_at_max.valuation, "Ask below cost"
        assert o.price <= seller_at_max.max_price, "Ask above max_price"
    for agent, label in [
        (buyer_at_min, "boundary buyer"),
        (seller_at_max, "boundary seller"),
    ]:
        o = agent.get_order(state)
        assert (
            agent.min_price <= o.price <= agent.max_price
        ), f"{label}: price {o.price} outside [min, max]"


def test_ziu_valuation_independence():
    state = MarketState(best_bid=None, best_ask=None, last_trade_price=None, tick=0)
    rng_state = random.getstate()
    random.seed(99)
    agent_low_val = ZIUAgent(
        agent_id=0, is_buyer=True, valuation=50.0, min_price=1, max_price=200
    )
    prices_low = [agent_low_val.get_order(state).price for _ in range(200)]
    random.seed(99)
    agent_high_val = ZIUAgent(
        agent_id=0, is_buyer=True, valuation=180.0, min_price=1, max_price=200
    )
    prices_high = [agent_high_val.get_order(state).price for _ in range(200)]
    random.setstate(rng_state)
    assert (
        prices_low == prices_high
    ), "ZI-U produced different prices for different valuations - valuation is leaking into draws"
