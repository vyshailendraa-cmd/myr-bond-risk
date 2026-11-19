"""One-page monthly portfolio factsheet as a PDF, generated straight from the engine."""
from __future__ import annotations

import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#1f3a5f")


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _table(data, widths, align_right_from=1):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def build_factsheet(out_path, curve, market, book, summary, stress, krd,
                    fund_name="Sample MYR Bond Portfolio", as_of: str | None = None,
                    commentary: str | None = None) -> str:
    as_of = as_of or date.today().isoformat()
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, textColor=NAVY, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading3"], fontSize=10, textColor=NAVY, spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8, leading=10.5)
    small = ParagraphStyle("s", parent=body, fontSize=6.5, textColor=colors.grey)

    # Charts
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    tt = np.linspace(0.1, 10.5, 200)
    ax.plot(tt, curve(tt) * 100, color="#1f3a5f", lw=2)
    ax.scatter(market["maturity_years"], market["yield_pct"], color="#111", s=14, zorder=5)
    ax.set_xlabel("Years", fontsize=7); ax.set_ylabel("Yield (%)", fontsize=7)
    ax.set_title("MGS yield curve", fontsize=8); ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
    curve_img = _png(fig)

    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.bar([f"{int(t)}y" for t in krd["key_tenor_y"]], krd["krd_dv01_rm"], color="#e67e22")
    ax.set_title("Key-rate DV01 (RM per bp)", fontsize=8); ax.tick_params(labelsize=6); ax.grid(axis="y", alpha=0.3)
    krd_img = _png(fig)

    story = [
        Paragraph(fund_name, h1),
        Paragraph(f"Risk factsheet as of {as_of}. Generated automatically from market data and holdings.", small),
        Spacer(1, 4),
    ]
    kpis = [["Market value", "Yield", "Mod. duration", "Convexity", "DV01"],
            [f"RM{summary['market_value_rm']/1e6:,.1f}m", f"{summary['yield_%']:.2f}%",
             f"{summary['mod_duration']:.2f}", f"{summary['convexity']:.1f}", f"RM{summary['dv01_rm']:,.0f}"]]
    story.append(_table(kpis, [36 * mm] * 5, align_right_from=0))
    story.append(Spacer(1, 4))
    story.append(Table([[Image(curve_img, 88 * mm, 50 * mm), Image(krd_img, 88 * mm, 50 * mm)]]))

    story.append(Paragraph("Holdings", h2))
    rows = [["Holding", "Type", "Maturity (y)", "Coupon %", "Yield %", "Price", "Weight %", "Mod. dur."]]
    for _, r in book.iterrows():
        rows.append([r["name"], r["type"], f"{r['maturity_y']:.2f}", f"{r['coupon_%']:.3f}", f"{r['yield_%']:.2f}",
                     f"{r['price']:.2f}", f"{r['weight_%']:.1f}", f"{r['mod_duration']:.2f}"])
    story.append(_table(rows, [52 * mm, 22 * mm, 17 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm], 2))

    story.append(Paragraph("OPR stress test (full revaluation)", h2))
    piv = stress.pivot(index="shock_bp", columns="scenario", values="pnl_rm")
    srows = [["OPR shock", "Parallel shift (RM)", "OPR pass-through (RM)"]]
    for s, r in piv.iterrows():
        srows.append([f"{s:+d}bp", f"{r['parallel']:,.0f}", f"{r['OPR pass-through']:,.0f}"])
    story.append(_table(srows, [30 * mm, 50 * mm, 50 * mm]))

    if commentary:
        story.append(Paragraph("Policy commentary", h2))
        story.append(Paragraph(commentary, body))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Illustrative portfolio for demonstration. Yields: Bank Negara Malaysia / Bursa Malaysia Bonds. "
        "Curve: Nelson-Siegel fit. OPR pass-through assumes the long end moves 40% as much as the OPR. "
        "Not investment advice.", small))

    SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                      topMargin=12 * mm, bottomMargin=12 * mm, title=f"{fund_name} factsheet").build(story)
    return str(out_path)
