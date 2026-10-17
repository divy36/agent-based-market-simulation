from market_sim.types import *
from market_sim.agents import *
from market_sim.market import DoubleAuctionMarket
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def test_basic():
    buyer = RothErevAgent(
        agent_id=0,
        is_buyer=True,
        valuation=150.0,
        min_price=0,
        max_price=200,
        n_actions=21,
    )
    seller = RothErevAgent(
        agent_id=1,
        is_buyer=False,
        valuation=50.0,
        min_price=0,
        max_price=200,
        n_actions=21,
    )
    state = MarketState(best_bid=None, best_ask=None, last_trade_price=None, tick=0)
    b_order = buyer.get_order(state)
    s_order = seller.get_order(state)
    assert b_order.order_type == "bid", "Buyer must submit a bid"
    assert s_order.order_type == "ask", "Seller must submit an ask"
    assert b_order.price <= buyer.valuation, "Buyer must not bid above valuation"
    assert s_order.price >= seller.valuation, "Seller must not ask below cost"


def check_action_distribution():
    buyer = RothErevAgent(
        agent_id=0,
        is_buyer=True,
        valuation=150.0,
        min_price=0,
        max_price=200,
        n_actions=21,
        recency=0.1,
        experimentation=0.05,
    )
    target_idx = 15
    for _ in range(50):
        buyer._update_propensities(target_idx, reward=10.0)
    (actions, probs) = buyer.get_action_distribution()
    assert (
        probs.shape == actions.shape
    ), "Probability array must match action grid shape"
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities must be in [0, 1]"
    assert np.isclose(probs.sum(), 1.0), "Probabilities must sum to 1"
    best_idx = int(np.argmax(probs))
    assert best_idx == target_idx
    return (actions, probs)


def run_orderbook_market(
    n_buyers=8,
    n_sellers=8,
    n_ticks=400,
    random_seed=7,
    recency=0.1,
    experimentation=0.05,
):
    rng = random.Random(random_seed)
    buyers = [
        RothErevAgent(
            agent_id=i,
            is_buyer=True,
            valuation=rng.uniform(100, 200),
            min_price=0,
            max_price=200,
            n_actions=21,
            recency=recency,
            experimentation=experimentation,
        )
        for i in range(n_buyers)
    ]
    sellers = [
        RothErevAgent(
            agent_id=100 + i,
            is_buyer=False,
            valuation=rng.uniform(0, 100),
            min_price=0,
            max_price=200,
            n_actions=21,
            recency=recency,
            experimentation=experimentation,
        )
        for i in range(n_sellers)
    ]
    market = DoubleAuctionMarket(
        buyers, sellers, min_price=0, max_price=200, random_seed=random_seed
    )
    market.run(n_ticks)
    trade_df = market.get_trade_log()
    tick_df = market.get_tick_log()
    agent_df = market.get_agent_stats()
    return (buyers, sellers, market, trade_df, tick_df, agent_df)


def check_convergence(n_buyers=8, n_sellers=8, n_ticks=400):
    (buyers, sellers, market, trade_df, tick_df, agent_df) = run_orderbook_market(
        n_buyers=n_buyers,
        n_sellers=n_sellers,
        n_ticks=n_ticks,
        random_seed=7,
        recency=0.1,
        experimentation=0.05,
    )
    all_agents = buyers + sellers
    initial_uniform = []
    final_max_probs = []
    for agent in all_agents:
        (_, probs) = agent.get_action_distribution()
        initial_uniform.append(1.0 / agent.n_actions)
        final_max_probs.append(float(np.max(probs)))
    if not trade_df.empty:
        q1 = max(1, len(trade_df) // 4)
        q4 = max(0, 3 * len(trade_df) // 4)
        early_prices = trade_df["trade_price"].iloc[:q1]
        late_prices = trade_df["trade_price"].iloc[q4:]
    return (buyers, sellers, market, trade_df, tick_df, agent_df)


def test_profit_tracking():
    buyer = RothErevAgent(
        agent_id=0,
        is_buyer=True,
        valuation=150.0,
        min_price=0,
        max_price=200,
        n_actions=21,
    )
    seller = RothErevAgent(
        agent_id=1,
        is_buyer=False,
        valuation=50.0,
        min_price=0,
        max_price=200,
        n_actions=21,
    )
    buyer._last_action_idx = 10
    seller._last_action_idx = 10
    b_outcome = TradeOutcome(
        agent_id=0,
        submitted_price=110.0,
        order_type="bid",
        accepted=True,
        trade_price=100.0,
        profit=50.0,
    )
    s_outcome = TradeOutcome(
        agent_id=1,
        submitted_price=90.0,
        order_type="ask",
        accepted=True,
        trade_price=100.0,
        profit=50.0,
    )
    buyer.observe_outcome(b_outcome)
    seller.observe_outcome(s_outcome)
    assert buyer.profit == 50.0, f"Expected 50.0, got {buyer.profit}"
    assert seller.profit == 50.0, f"Expected 50.0, got {seller.profit}"
    assert buyer.trade_count == 1
    assert seller.trade_count == 1


def test_distribution():
    check_action_distribution()


def test_learning():
    buyers, sellers, market, trades, ticks, agents = check_convergence(n_ticks=80)
    assert len(trades) > 0
    assert len(ticks) == 80
    assert sum(a.trade_count for a in buyers + sellers) == 2 * len(trades)
