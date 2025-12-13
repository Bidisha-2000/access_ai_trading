from __future__ import annotations

from omago_ai.orchestrator.parsing import parse_trade_request
from omago_ai.orchestrator.router import LeadRouter
from omago_ai.orchestrator.state import SessionState


def test_parse_trade_request_basic():
    tr = parse_trade_request("buy $200 of TSLA")
    assert tr is not None
    assert tr.side == "buy"
    assert tr.ticker == "TSLA"
    assert tr.notional_usd == 200.0


def test_router_trade_flow_produces_risk_and_synthesis():
    router = LeadRouter()
    res = router.handle("buy $200 of TSLA",
                        session=SessionState(reading_level="simple"))
    assert res.intent == "trade"
    assert res.risk is not None
    assert res.synthesis is not None
    assert "Trade check" in res.final_message
