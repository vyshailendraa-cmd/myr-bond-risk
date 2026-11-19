"""AI summary of a Bank Negara Monetary Policy Statement (MPS), linked to portfolio risk.

Design rule: the language model only READS the statement (stance, decision,
key drivers). Every number about the portfolio comes from the risk engine,
so the model cannot invent figures.

Needs ANTHROPIC_API_KEY in the environment. Model can be changed with
the MPS_MODEL environment variable.
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_MODEL = os.environ.get("MPS_MODEL", "claude-sonnet-5-5")

PROMPT = """You are a fixed-income analyst at a Malaysian fund house.
Read the Bank Negara Malaysia Monetary Policy Statement below and reply with
JSON only, using exactly these keys:

{{
  "opr_decision": "hold" | "hike" | "cut",
  "opr_level_pct": number or null,
  "change_bp": number,
  "stance": "hawkish" | "neutral" | "dovish",
  "key_points": [up to 4 short strings on growth, inflation and the ringgit],
  "risks": [up to 3 short strings],
  "next_move_bias": "hike" | "hold" | "cut"
}}

Use only what the statement says. Do not add any numbers that are not in it.

STATEMENT:
{text}
"""


def fetch_text(source: str) -> str:
    """Read a statement from a local text file or a URL (HTML is stripped)."""
    if os.path.exists(source):
        return open(source, encoding="utf-8").read()
    import httpx
    html = httpx.get(source, timeout=30, follow_redirects=True).text
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("Model did not return JSON")
    return json.loads(m.group(0))


def read_statement(text: str, client=None, model: str = DEFAULT_MODEL) -> dict:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": PROMPT.format(text=text[:20000])}],
    )
    raw = "".join(getattr(b, "text", "") for b in msg.content)
    out = _parse_json(raw)
    allowed = {"hold", "hike", "cut"}
    if out.get("opr_decision") not in allowed or out.get("next_move_bias") not in allowed:
        raise ValueError(f"Unexpected values from model: {out}")
    return out


def portfolio_impact(reading: dict, stress_table) -> str:
    """Deterministic sentence linking the statement to the portfolio.

    stress_table is the output of stress.run_scenarios().
    """
    bias = reading["next_move_bias"]
    if bias == "hold":
        return ("BNM signals no change ahead. The main rate risk is a surprise move: "
                + _line(stress_table, 25) + " " + _line(stress_table, -25))
    shock = 25 if bias == "hike" else -25
    return f"BNM's bias is towards a {bias}. " + _line(stress_table, shock)


def _line(st, shock_bp: int) -> str:
    row = st[(st["shock_bp"] == shock_bp) & (st["scenario"] == "OPR pass-through")].iloc[0]
    sign = "+" if row["pnl_rm"] >= 0 else "-"
    return (f"A {shock_bp:+d}bp OPR move would change portfolio value by "
            f"{sign}RM{abs(row['pnl_rm']):,.0f} ({row['pnl_%']:+.2f}%).")
