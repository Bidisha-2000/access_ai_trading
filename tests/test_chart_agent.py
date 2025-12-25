from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from omago_ai.agents.chart_agent import ChartAgent


def test_chart_agent_explains_basic_series():
    prices = pd.DataFrame(
        {
            "ts": [
                datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 1, 0, 2, tzinfo=timezone.utc),
            ],
            "price": [100.0, 101.0, 102.0],
        }
    )

    res = ChartAgent().explain(ticker="TSLA", prices=prices, reading_level="simple")
    assert "Start price" in res.final_message
    assert "End price" in res.final_message
    assert "simulated" in res.final_message.lower()


def test_chart_agent_handles_empty():
    prices = pd.DataFrame(columns=["ts", "price"])
    res = ChartAgent().explain(ticker="AAPL", prices=prices, reading_level="simple")
    assert "no price data" in res.final_message.lower()
