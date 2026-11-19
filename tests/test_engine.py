from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from bondrisk.curve import fit_nelson_siegel
from bondrisk.data import parse_gov_sec_yield, simulate_history
from bondrisk.forecast import extract_factors, to_weekly
from bondrisk.mps import portfolio_impact, read_statement
from bondrisk.portfolio import load_holdings, portfolio_summary, value_portfolio
from bondrisk.stress import SHAPE_SCENARIOS, key_rate_durations, run_scenarios, scenario_pnl


@pytest.fixture(scope="module")
def setup():
    m = pd.read_csv("data/mgs_curve_2026-11-18.csv")
    curve, rmse = fit_nelson_siegel(m["maturity_years"], m["yield_pct"] / 100)
    return curve, rmse, load_holdings("data/sample_portfolio.csv")


def test_curve_fits_market_within_10bp(setup):
    _, rmse, _ = setup
    assert rmse < 10


def test_key_rate_dv01s_sum_to_parallel_dv01(setup):
    curve, _, h = setup
    krd = key_rate_durations(h, curve)["krd_dv01_rm"].sum()
    dv01 = portfolio_summary(value_portfolio(h, curve))["dv01_rm"]
    assert krd == pytest.approx(dv01, rel=0.01)


def test_convexity_makes_cuts_gain_more_than_hikes_lose(setup):
    curve, _, h = setup
    st = run_scenarios(h, curve)
    p = st[st.scenario == "parallel"].set_index("shock_bp")["pnl_rm"]
    assert p[-100] > -p[100] > 0


def test_bear_scenarios_lose_bull_scenarios_gain(setup):
    curve, _, h = setup
    df = scenario_pnl(h, curve, SHAPE_SCENARIOS).set_index("scenario")["pnl_rm"]
    assert all(df[k] < 0 for k in df.index if k.startswith("Bear"))
    assert all(df[k] > 0 for k in df.index if k.startswith("Bull"))


def test_bnm_parser_handles_nested_json():
    payload = {"data": [
        {"mgs_security_type": "MGS", "tenure": "3-Year", "closing": "3.40"},
        {"mgs_security_type": "MGS", "tenure": "10 Year", "closing": 3.83},
        {"mgs_security_type": "GII", "tenure": "10-Year", "closing": 3.86},
    ]}
    assert parse_gov_sec_yield(payload) == {3: 3.40, 10: 3.83}


def test_factor_extraction_matches_curve_level():
    F = extract_factors(to_weekly(simulate_history(300)))
    assert F["level"].between(0.02, 0.07).all()


def _fake_client(json_text):
    msg = SimpleNamespace(content=[SimpleNamespace(text=json_text)])
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: msg))


def test_mps_reading_and_impact_use_engine_numbers(setup):
    curve, _, h = setup
    fake = _fake_client('{"opr_decision":"hold","opr_level_pct":2.75,"change_bp":0,"stance":"neutral",'
                        '"key_points":["Growth steady"],"risks":[],"next_move_bias":"hike"}')
    reading = read_statement("statement text", client=fake)
    text = portfolio_impact(reading, run_scenarios(h, curve))
    assert "hike" in text and "+25bp" in text and "-RM" in text


def test_mps_rejects_bad_model_output():
    with pytest.raises(ValueError):
        read_statement("x", client=_fake_client('{"opr_decision":"maybe","next_move_bias":"hold"}'))


def test_api_endpoints():
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/portfolio").json()["summary"]["mod_duration"] > 0
    r = c.get("/stress/opr-shock?bp=25").json()["results"]
    assert all(x["pnl_rm"] < 0 for x in r)
    assert c.get("/factsheet.pdf").content[:4] == b"%PDF"
