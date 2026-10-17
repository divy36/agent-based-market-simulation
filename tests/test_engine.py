import numpy as np
import pytest
from market_sim.types import Order
from market_sim.order_book import OrderBook
from market_sim.market import DoubleAuctionMarket
from market_sim.pipeline import SimulationPipeline
from market_sim.agents import ZICAgent, ZIUAgent, GDAgent, RothErevAgent
from market_sim.statistics import holm_adjust


def test_price_time_priority():
    book = OrderBook()
    book.submit_order(Order(1, "ask", 90))
    book.submit_order(Order(2, "ask", 90))
    book.submit_order(Order(3, "ask", 80))
    assert [
        book.submit_order(Order(10 + i, "bid", 100))[0].seller_id for i in range(3)
    ] == [3, 1, 2]


@pytest.mark.parametrize(
    "price,quantity", [(float("nan"), 1), (float("inf"), 1), (-1, 1), (10, 2)]
)
def test_invalid_order(price, quantity):
    with pytest.raises(ValueError):
        OrderBook().submit_order(Order(1, "bid", price, quantity))


@pytest.mark.parametrize("agent", [ZICAgent, ZIUAgent, GDAgent, RothErevAgent])
def test_accounting_and_reproducibility(agent):

    def run():
        buyers = [agent(i, True, 150, 0, 200) for i in range(4)]
        sellers = [agent(i + 4, False, 50, 0, 200) for i in range(4)]
        market = DoubleAuctionMarket(buyers, sellers, random_seed=42)
        market.run(40)
        return market

    market = run()
    trades = market.get_trade_log()
    assert len(trades) > 0 or agent is GDAgent
    assert sum((a.profit for a in market.all_agents)) == pytest.approx(
        len(trades) * 100
    )
    assert sum((a.trade_count for a in market.all_agents)) == len(trades) * 2
    assert SimulationPipeline(market).allocative_efficiency() == pytest.approx(
        len(trades) / 160 * 100
    )
    assert trades.equals(run().get_trade_log())


def test_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        DoubleAuctionMarket(
            [ZICAgent(0, True, 100, 0, 200)], [ZICAgent(0, False, 50, 0, 200)]
        )


def test_persistent_orders_rejected():
    with pytest.raises(ValueError, match="clearing"):
        DoubleAuctionMarket([], [], clear_book_each_tick=False)


def test_holm_monotone():
    assert np.allclose(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])


def test_empty_efficiency():
    assert SimulationPipeline(DoubleAuctionMarket([], [])).allocative_efficiency() == 0


def test_gd_belief_matches_reference():
    from market_sim.types import HistoryRecord

    agent = GDAgent(0, True, 150, 0, 200, price_steps=20)
    agent.history = [
        HistoryRecord(float(i * 9), "bid" if i % 2 else "ask", i % 3 == 0)
        for i in range(20)
    ]
    expected = []
    for i in range(agent.price_steps):
        alpha = beta = 0
        for record in agent.history:
            index = int(np.argmin(abs(agent.price_grid - record.price)))
            if (
                record.order_type == "bid"
                and index >= i
                or (record.order_type == "ask" and index <= i)
            ):
                if record.accepted:
                    alpha += 1
                else:
                    beta += 1
        expected.append((alpha + 1) / (alpha + beta + 2))
    assert np.allclose(agent._compute_belief(), expected)
