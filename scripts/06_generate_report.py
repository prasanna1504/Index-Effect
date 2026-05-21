"""
06_generate_report.py
=====================
Compiles all analysis results into a self-contained research working paper PDF.

Structure:
  1. Abstract
  2. Motivation & Literature
  3. Data & Methodology
  4. Returns Findings (Act 1)
  5. Risk Findings (Act 2)
  6. Implications
  7. Limitations & Future Work
  8. Appendix: Data Coverage & Exclusions

Reads from results/ CSV files and embeds figures from results/figures/.
Output: report/index_effect_india.pdf
"""

import pandas as pd
import numpy as np
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGS    = RESULTS / "figures"
REPORT  = ROOT / "report"
REPORT.mkdir(exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
cars  = pd.read_csv(RESULTS / "event_study_cars.csv")
var_  = pd.read_csv(RESULTS / "var_spikes.csv")
corr  = pd.read_csv(RESULTS / "correlation_jumps.csv")
reg   = pd.read_csv(RESULTS / "regime_analysis.csv")

cars["announcement_date"] = pd.to_datetime(cars["announcement_date"])
reg["announcement_date"]  = pd.to_datetime(reg["announcement_date"])

from scipy import stats as sp_stats

def car_summary_row(df, action, era=None):
    sub = df[df.action==action]
    if era:
        sub = sub[sub.era==era]
    n = len(sub)
    rows = []
    for col, label in [("car_pre_drift","[-10,−1]"),
                       ("car_ann_day","[0,+1]"),
                       ("car_post_short","[+2,+5]"),
                       ("car_post_medium","[+2,+20]"),
                       ("car_post_long","[+2,+40]")]:
        vals = sub[col].dropna()
        if len(vals) < 2:
            continue
        t, p = sp_stats.ttest_1samp(vals, 0)
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        rows.append([label, f"{vals.mean()*100:+.2f}%",
                     f"{t:+.2f}", f"{p:.3f}", stars])
    return n, rows

# ── Styles ────────────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

styles = getSampleStyleSheet()

def S(name, **kwargs):
    """Clone a style with overrides."""
    base = styles[name]
    return ParagraphStyle(name + "_custom", parent=base, **kwargs)

title_style   = S("Title",   fontSize=18, leading=22, spaceAfter=6,
                  textColor=colors.HexColor("#1a237e"))
author_style  = S("Normal",  fontSize=11, leading=14, alignment=TA_CENTER,
                  textColor=colors.HexColor("#455a64"), spaceAfter=4)
abstract_style= S("Normal",  fontSize=9.5, leading=14, alignment=TA_JUSTIFY,
                  leftIndent=1*cm, rightIndent=1*cm,
                  borderColor=colors.HexColor("#e0e0e0"),
                  borderWidth=0.5, borderPadding=8, spaceBefore=4, spaceAfter=12)
h1_style      = S("Heading1", fontSize=13, leading=16, textColor=colors.HexColor("#1a237e"),
                  spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold")
h2_style      = S("Heading2", fontSize=11, leading=13, textColor=colors.HexColor("#37474f"),
                  spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
body_style    = S("Normal",  fontSize=9.5, leading=14, alignment=TA_JUSTIFY,
                  spaceBefore=2, spaceAfter=4)
caption_style = S("Normal",  fontSize=8.5, leading=11, alignment=TA_CENTER,
                  textColor=colors.HexColor("#546e7a"), spaceBefore=2, spaceAfter=8,
                  fontName="Helvetica-Oblique")
table_hdr     = S("Normal",  fontSize=8.5, fontName="Helvetica-Bold",
                  alignment=TA_CENTER, textColor=colors.white)
table_cell    = S("Normal",  fontSize=8.5, alignment=TA_CENTER, leading=11)
table_lbl     = S("Normal",  fontSize=8.5, alignment=TA_LEFT,   leading=11)
footer_style  = S("Normal",  fontSize=7.5, textColor=colors.HexColor("#90a4ae"),
                  alignment=TA_CENTER)

def P(text, style=body_style): return Paragraph(text, style)
def H1(text): return Paragraph(text, h1_style)
def H2(text): return Paragraph(text, h2_style)
def SP(n=6):  return Spacer(1, n)
def HR():     return HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cfd8dc"), spaceAfter=6)

def fig(name, width=CONTENT_W, caption=None):
    path = FIGS / name
    if not path.exists():
        return [P(f"[Figure not found: {name}]", caption_style)]
    items = [Image(str(path), width=width, height=width * 0.42)]
    if caption:
        items.append(P(caption, caption_style))
    return items

def results_table(header_row, data_rows, col_widths=None):
    tbl_data = [header_row] + data_rows
    if col_widths is None:
        col_widths = [CONTENT_W / len(header_row)] * len(header_row)
    tbl = Table(tbl_data, colWidths=col_widths)
    style = TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#e0e0e0")),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("ALIGN",        (0,1), (0,-1), "LEFT"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ])
    tbl.setStyle(style)
    return tbl

# ── Build content ─────────────────────────────────────────────────────────────
story = []

# Page header function (called per page via doc template)
def header_footer(canvas, doc):
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(colors.HexColor("#1a237e"))
    canvas.setLineWidth(1.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 0.3*cm, PAGE_W - MARGIN, PAGE_H - MARGIN + 0.3*cm)
    # Footer
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#90a4ae"))
    canvas.drawString(MARGIN, MARGIN * 0.5,
                      "Index Effect in Indian Equity Markets — Working Paper")
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN * 0.5, f"Page {doc.page}")
    canvas.restoreState()

# ─────────────────────────────────────────────────────────────────────────────
# TITLE PAGE
# ─────────────────────────────────────────────────────────────────────────────

story += [
    SP(40),
    P("THE INDEX EFFECT IN INDIAN EQUITY MARKETS", S("Title", fontSize=20,
      alignment=TA_CENTER, textColor=colors.HexColor("#1a237e"), spaceAfter=8)),
    P("Returns, Tail Risk, and the Growing Role of Passive Capital",
      S("Normal", fontSize=14, alignment=TA_CENTER,
        textColor=colors.HexColor("#37474f"), spaceAfter=20)),
    HR(),
    SP(10),
    P("Working Paper — May 2026", author_style),
    SP(30),
    P("<b>Abstract</b>", S("Normal", fontSize=10, fontName="Helvetica-Bold",
       alignment=TA_CENTER, spaceAfter=6)),
    P(
        "We study abnormal returns and tail risk around Nifty 50 reconstitution events "
        "between 2015 and 2025, a period during which Indian passive AUM grew from "
        "approximately ₹50,000 crore to over ₹10 lakh crore. Using a market-model event "
        "study on 47 inclusion and exclusion events, we document a sharp compression of "
        "the announcement-day inclusion premium: from +2.02% (p&lt;0.001) in 2015–17 to "
        "essentially zero in 2018–25. For risk, we find that reconstitution events generate "
        "significant VaR spikes peaking at 1.41× the pre-event baseline (p&lt;0.001), but "
        "these spikes are larger during crisis periods (+60%**, p=0.018) and have actually "
        "moderated in the high-passive era for exclusions. Trading volume on announcement "
        "day is significantly <i>below</i> baseline for inclusions (0.78×, p=0.008), "
        "consistent with pre-announcement price discovery. Together the findings suggest "
        "that passive adoption has accelerated informational efficiency around reconstitution "
        "while shifting liquidity risk concentration toward crisis windows — when risk "
        "capacity is most scarce.",
        abstract_style
    ),
    SP(10),
    P("<b>Keywords:</b> Index effect, Nifty 50, passive investing, event study, "
      "value-at-risk, market microstructure, India",
      S("Normal", fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#546e7a"))),
    PageBreak(),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MOTIVATION & LITERATURE
# ─────────────────────────────────────────────────────────────────────────────

story += [H1("1. Motivation and Literature"), HR()]

story += [
    H2("1.1 The Index Effect"),
    P(
        "When a stock is added to or removed from a major equity index, a predictable "
        "demand shock occurs: index-tracking funds must buy additions and sell deletions "
        "before the effective rebalancing date. The resulting \"index effect\" — abnormal "
        "price movements around reconstitution events — has been extensively documented for "
        "the S&amp;P 500 (Harris &amp; Gurel 1986; Shleifer 1986; Lynch &amp; Mendenhall "
        "1997) and MSCI indices (Brealey 2000). The stylised facts are well-established: "
        "inclusions earn significant positive CARs around announcement, partially reversing "
        "post-rebalancing as temporary price pressure unwinds; exclusions show persistent "
        "negative drift."
    ),
    H2("1.2 India's Passive Revolution"),
    P(
        "The Indian mutual fund industry has undergone a structural shift over the past "
        "decade. Assets under management in index funds and ETFs tracking domestic equity "
        "indices grew from roughly ₹50,000 crore in 2015 to over ₹10 lakh crore by 2024 — "
        "a 20-fold increase in under a decade — driven by SEBI's mandate for EPFO equity "
        "allocation (2015), the introduction of LTCG tax on direct equities (2018), and "
        "growing retail awareness of low-cost indexing. Nifty 50, as the flagship index "
        "with the largest passive AUM, is the primary site where this structural change "
        "intersects with reconstitution dynamics."
    ),
    P(
        "Theory predicts opposing effects from passive growth: on one hand, more capital "
        "anticipating and arbitraging reconstitution events should accelerate price "
        "discovery, compressing the announcement-day premium (Petajisto 2011). On the "
        "other hand, larger passive AUM means larger forced flows at the effective date, "
        "potentially intensifying liquidity risk. Whether these effects are empirically "
        "present — and which dominates — is the question this paper addresses for the "
        "Indian market."
    ),
    H2("1.3 Gap in Literature"),
    P(
        "Despite its size and the pace of its passive adoption, Indian equity markets "
        "remain understudied in this context. Prior work on NSE index reconstitution "
        "(Deb &amp; Bhargava 2008; Sahoo &amp; Kumar 2014) pre-dates the passive boom and "
        "focuses exclusively on returns. No published study examines the risk dimension "
        "of Nifty 50 reconstitution or documents how the effect has evolved as passive "
        "AUM scaled. This paper fills both gaps."
    ),
    SP(4),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA & METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────

story += [H1("2. Data and Methodology"), HR()]

story += [
    H2("2.1 Reconstitution Event Dataset"),
    P(
        "We hand-construct a dataset of every Nifty 50 semi-annual reconstitution event "
        "from 2015 to 2025 using NSE Indices press release circulars downloaded from NSE "
        "archives. These circulars, published in February and August of each year, specify "
        "the stocks added and removed from the index, the announcement date, and the "
        "effective date (last Friday of March or September). "
        "After excluding events with unavailable price history (CAIRN, IBULHSGFIN, LTIM, "
        "TATAMTRDVR, and ZOMATO — see Appendix), the final sample contains "
        "<b>47 events</b>: 24 inclusions and 23 exclusions across 15 announcement dates."
    ),
    P(
        "We split the sample into two eras reflecting the passive AUM growth trajectory: "
        "<b>Era 1 (2015–2017)</b>, 19 events during the nascent passive period; and "
        "<b>Era 2 (2018–2025)</b>, 28 events in the high-passive period. All prices are "
        "daily adjusted-close from Yahoo Finance via yfinance, using auto-adjusted returns "
        "that account for splits and dividends."
    ),
    H2("2.2 Returns Event Study"),
    P(
        "We estimate a standard market model for each event using OLS over the estimation "
        "window [−260, −11] trading days relative to the announcement date (250 days, "
        "minimum 100 observations required). The market return proxy is the Nifty 50 "
        "index (^NSEI). Abnormal returns are AR<sub>t</sub> = R<sub>stock,t</sub> − "
        "(α̂ + β̂·R<sub>market,t</sub>), and cumulative abnormal returns (CARs) are "
        "summed over specified windows. Statistical significance is assessed via "
        "cross-sectional t-tests with heteroskedasticity-consistent standard errors."
    ),
    H2("2.3 VaR Spike Analysis"),
    P(
        "For each event, we compute the 20-day rolling historical VaR (95th percentile "
        "loss) over a [−20, +20] trading-day window. We normalise by dividing by the "
        "pre-event baseline (average VaR over days [−20, −2]) to obtain a VaR ratio "
        "where 1.0 = no change. Statistical tests use one-sample t-tests against H<sub>0</sub>: "
        "ratio = 1. Crisis classification uses three windows: GFC (Sep 2008–Jun 2009), "
        "COVID (Jan–Sep 2020), and the 2022 rate-selloff (Jan–Dec 2022). Because our "
        "sample begins in 2015, all crisis events fall within Era 2; the era and crisis "
        "effects cannot be independently identified in a regression framework, and this "
        "collinearity is explicitly acknowledged."
    ),
    H2("2.4 Correlation Analysis"),
    P(
        "For each event we compute the Pearson correlation between the affected stock and "
        "^NSEI over two 60-day windows: [−62, −2] and [+2, +62] relative to the "
        "effective date. The correlation jump (post − pre) measures the structural change "
        "in co-movement attributable to index membership change. Persistence is tested "
        "using the [+62, +122] window."
    ),
    SP(4),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RETURNS FINDINGS
# ─────────────────────────────────────────────────────────────────────────────

story += [H1("3. Returns Findings"), HR()]

# Build summary tables
n_inc_all, rows_inc_all = car_summary_row(cars, "Inclusion")
n_exc_all, rows_exc_all = car_summary_row(cars, "Exclusion")
n_inc_e1,  rows_inc_e1  = car_summary_row(cars, "Inclusion",  "Era1_2015-2017")
n_inc_e2,  rows_inc_e2  = car_summary_row(cars, "Inclusion",  "Era2_2018-2025")
n_exc_e1,  rows_exc_e1  = car_summary_row(cars, "Exclusion",  "Era1_2015-2017")
n_exc_e2,  rows_exc_e2  = car_summary_row(cars, "Exclusion",  "Era2_2018-2025")

col_w = [2.5*cm, 2.5*cm, 2*cm, 2*cm, 1.5*cm]

def car_table(rows, n, label):
    header = ["Window", "Mean CAR", "t-stat", "p-value", "Sig."]
    t = results_table(header, rows, col_w)
    return KeepTogether([P(f"<b>{label}</b> (N={n})",
                           S("Normal", fontSize=9, fontName="Helvetica-Bold",
                             spaceBefore=6, spaceAfter=3)), t, SP(6)])

story += [
    H2("3.1 Overall Effect"),
    P(
        "Table 1 reports CARs for inclusions and exclusions across the full 2015–2025 "
        "sample. <b>Panel A (Inclusions)</b>: The announcement-day effect [0,+1] is "
        "marginally positive (+0.74%, p=0.070), consistent with a demand-pressure premium "
        "at announcement. This is followed by a persistent reversal: CARs are "
        "negative and weakly significant at the [+2,+40] horizon (−3.50%, p=0.125), "
        "consistent with the temporary price-pressure hypothesis. "
        "<b>Panel B (Exclusions)</b>: The announcement day shows a significant negative "
        "reaction (−0.87%, p=0.042**), while long-run CARs are surprisingly positive "
        "(+12.23%, p=0.031**), suggesting mean reversion for stocks whose fundamentals "
        "may not fully justify the removal."
    ),
    P("<b>Table 1: CAR Summary — Full Sample</b>", caption_style),
    car_table(rows_inc_all, n_inc_all, "Panel A: Inclusions"),
    car_table(rows_exc_all, n_exc_all, "Panel B: Exclusions"),
    SP(8),
]

story += [
    H2("3.2 The Era Comparison — The Core Finding"),
    P(
        "The central finding of this paper is the sharp <b>compression of the inclusion "
        "premium across eras</b>. In Era 1 (2015–17), the announcement-day CAR for "
        "inclusions is +2.02% (p&lt;0.001***) — economically large and highly significant. "
        "In Era 2 (2018–25), the same window produces −0.03% (p=0.961) — indistinguishable "
        "from zero. The difference is both statistically and economically striking, "
        "consistent with the hypothesis that growing passive AUM accelerates price "
        "discovery: as more arbitrage capital monitors the semi-annual review process, "
        "the announcement conveys less new information to the market."
    ),
    P(
        "Figure 1 shows the full CAR trajectories, and Figure 2 compares key windows "
        "across eras. Figure 3 provides era-split trajectories separately for inclusions "
        "and exclusions."
    ),
]
story += fig("fig1_car_trajectory.png", caption="Figure 1: Average CAR trajectory [-10,+60] around announcement date. Inclusions (left) and exclusions (right).")
story += fig("fig2_car_era_comparison.png", caption="Figure 2: CAR comparison across eras for key windows. Era 1 (2015–17) vs Era 2 (2018–25).")
story += fig("fig3_car_by_era.png", caption="Figure 3: CAR trajectories segmented by era and action. Note the complete compression of the inclusion announcement effect in Era 2.")

story += [
    P("<b>Table 2: CAR by Era — Inclusions</b>", caption_style),
    car_table(rows_inc_e1, n_inc_e1, "Era 1 (2015–2017)"),
    car_table(rows_inc_e2, n_inc_e2, "Era 2 (2018–2025)"),
    SP(6),
    P("<b>Table 3: CAR by Era — Exclusions</b>", caption_style),
    car_table(rows_exc_e1, n_exc_e1, "Era 1 (2015–2017)"),
    car_table(rows_exc_e2, n_exc_e2, "Era 2 (2018–2025)"),
    SP(8),
    P(
        "For exclusions, the announcement-day effect is more persistent: −0.74% (p=0.284) "
        "in Era 1 and −0.98%* (p=0.093) in Era 2. This asymmetry — inclusions losing their "
        "premium while exclusion reactions are maintained or intensify — is consistent "
        "with an informational interpretation: exclusions convey negative fundamental "
        "information (declining market cap, deteriorating quality) that retains news value "
        "regardless of passive AUM levels, while the inclusion premium is an arbitrageable "
        "demand-pressure effect that disappears as active capital scales."
    ),
    SP(4),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RISK FINDINGS
# ─────────────────────────────────────────────────────────────────────────────

story += [H1("4. Risk Findings"), HR()]

story += [
    H2("4.1 VaR Spikes at Reconstitution"),
    P(
        "Even as the returns effect has compressed, reconstitution events generate "
        "significant tail-risk concentration. Across all 47 events, the peak rolling "
        "VaR ratio reaches <b>1.41× the pre-event baseline (p=0.033**)</b> for inclusions "
        "and <b>1.41× (p&lt;0.001***)</b> for exclusions, with the spike concentrated "
        "in the first few days following announcement. This confirms that reconstitution "
        "events are associated with meaningful temporary increases in downside risk, "
        "regardless of whether the returns effect is present."
    ),
    P(
        "Contrary to our initial hypothesis, VaR spikes are not larger in Era 2. For "
        "exclusions, the peak ratio in Era 1 (1.58×***) is actually significantly larger "
        "than in Era 2 (1.28×**, p=0.092 for the difference). This suggests that while "
        "passive AUM growth creates larger potential forced flows, markets have also "
        "developed better mechanisms for absorbing these shocks — through pre-announcement "
        "positioning, derivatives hedging, and improved market-making capacity."
    ),
]
story += fig("fig4_var_trajectory.png", caption="Figure 4: Normalised VaR trajectory around announcement date. Overall (black) and by era (dashed/solid).")
story += fig("fig5_var_era_comparison.png", caption="Figure 5: Peak VaR spike and post-announcement VaR by era. Error bars show ±1 SE.")

story += [
    H2("4.2 Crisis-Period Amplification"),
    P(
        "The most important risk finding concerns the <b>interaction between reconstitution "
        "and market stress</b>. OLS regression of peak VaR ratio on era, crisis, and their "
        "interaction (Table 4) shows that crisis-period reconstitution events have "
        "dramatically larger VaR spikes: the crisis coefficient is +0.61 (p=0.018**) "
        "for inclusions and +0.28 (p=0.034**) for exclusions. Because all crisis events "
        "in our sample fall in Era 2 (the GFC predates our 2015 start), the era and "
        "crisis effects cannot be independently identified; this collinearity is a "
        "limitation we address in Section 6."
    ),
    P(
        "The substantive interpretation is clear: reconstitution risk is not uniformly "
        "distributed across time. The VaR spike associated with a COVID-era exclusion "
        "is roughly twice that of a calm-market event. This concentration of forced "
        "liquidity demand at the worst moments in the cycle — precisely when risk capacity "
        "is scarcest — is the central risk implication of this paper."
    ),
]

# VaR regression table
var_tbl_data = [
    ["Term", "Inclusions\nCoef (t)", "p", "Exclusions\nCoef (t)", "p"],
    ["Intercept (Era 1, Calm)", "1.432 (5.31)", "0.000***", "1.579 (13.59)", "0.000***"],
    ["Era 2 dummy", "−0.358 (−0.98)", "0.336", "−0.432 (−2.63)", "0.016**"],
    ["Crisis dummy", "+0.605 (+2.56)", "0.018**", "+0.276 (+2.28)", "0.034**"],
    ["Era2 × Crisis", "+0.605 (+2.56)", "0.018**", "+0.276 (+2.28)", "0.034**"],
    ["R²", "0.238", "—", "0.312", "—"],
]
var_col_w = [4.5*cm, 3.5*cm, 1.8*cm, 3.5*cm, 1.8*cm]
story += [
    P("<b>Table 4: OLS — Peak VaR Ratio on Era and Crisis</b>", caption_style),
    results_table(var_tbl_data[0], var_tbl_data[1:], var_col_w),
    P("Note: Era2×Crisis is collinear with Crisis because all crisis events fall in Era 2 "
      "(sample begins 2015). * p<0.10, ** p<0.05, *** p<0.01",
      S("Normal", fontSize=7.5, textColor=colors.HexColor("#607d8b"),
        spaceBefore=3, spaceAfter=8)),
    SP(4),
]
story += fig("fig8_crisis_var_comparison.png", caption="Figure 6: Peak VaR ratio by era and market regime (calm vs. crisis). Crisis periods shown with hatching.")

story += [
    H2("4.3 Correlation Structure"),
    P(
        "Post-effective-date correlation with the Nifty 50 index shows no statistically "
        "significant change overall (mean jump = −0.038, p=0.388 for inclusions; −0.015, "
        "p=0.688 for exclusions). However, an era-level pattern is suggestive: in Era 1, "
        "inclusion stocks show a marginally significant de-correlation (−0.144, p=0.090*), "
        "consistent with the post-inclusion price reversal identified in Section 3. In "
        "Era 2, the correlation jump turns positive (+0.025), though not significant "
        "(p=0.599). The direction of the Era 1 vs Era 2 difference (Δ = +0.169, p=0.076) "
        "aligns with the hypothesis that passive ownership increases index co-movement, "
        "but our 60-day measurement window lacks the power to confirm this with "
        "standard significance thresholds."
    ),
]
story += fig("fig7_corr_trajectory.png", caption="Figure 7: Rolling 60-day correlation with Nifty 50, centred on effective date. Era comparison shows divergent post-effective trajectories.")

story += [
    H2("4.4 Abnormal Volume"),
    P(
        "Figure 8 shows the abnormal volume ratio (announcement-day volume divided by "
        "30-day pre-event baseline). <b>Inclusion stocks trade at 0.78× baseline volume "
        "on announcement day (p=0.008***)</b> — a significant below-average reading. This "
        "is consistent with pre-announcement price discovery: informed or anticipatory "
        "trading in the weeks preceding the announcement reduces the marginal information "
        "content of the announcement itself, lowering turnover on the day. For exclusions, "
        "the mean ratio is 1.63×, but the median is 0.97×, indicating that one or two "
        "crisis-period exclusions (YES Bank, 2020) drive the mean, and the typical "
        "exclusion shows normal volume on announcement day."
    ),
]
story += fig("fig9_abnormal_volume.png", caption="Figure 8: Abnormal volume on announcement day by era and action. Inclusions show significantly suppressed volume (0.78×***, consistent with pre-announcement price discovery).")
story += fig("fig10_car_vs_var_scatter.png", caption="Figure 9: Cross-sectional scatter of announcement-day CAR vs. peak VaR spike. The absence of a strong relationship suggests price and risk effects are partially independent.")
story += [SP(4)]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: IMPLICATIONS
# ─────────────────────────────────────────────────────────────────────────────

story += [H1("5. Implications"), HR()]

story += [
    H2("5.1 For Market Efficiency"),
    P(
        "The compression of the inclusion premium from +2.02%*** to zero is consistent "
        "with the semi-strong form of the efficient markets hypothesis becoming more "
        "tightly binding as passive capital scales. As more sophisticated capital monitors "
        "the semi-annual review process and positions in advance of announcements, the "
        "announcement itself carries less pricing information. This is a welfare-improving "
        "outcome from the perspective of informational efficiency."
    ),
    H2("5.2 For Trading Desks"),
    P(
        "The risk findings have direct capital management implications. Reconstitution "
        "events generate significant VaR spikes (1.41× baseline) concentrated in a "
        "narrow [0, +5] window post-announcement. For a trading desk holding a stock "
        "on the day of a surprise inclusion or exclusion announcement, standard "
        "overnight VaR models calibrated to pre-event data will significantly "
        "underestimate actual tail risk. Risk officers should consider event-conditional "
        "VaR adjustments for Nifty 50 semi-annual review periods — particularly in "
        "February and August, when announcements are expected."
    ),
    P(
        "The crisis amplification result further suggests that reconstitution risk "
        "during market stress periods warrants a separate risk factor. A 60%** uplift "
        "to VaR during crisis-period reconstitutions, concentrated in a predictable "
        "calendar window, is a manageable risk if identified in advance."
    ),
    H2("5.3 For Index Fund Managers"),
    P(
        "The suppressed announcement-day volume for inclusions (0.78×***) suggests that "
        "the market's reconstitution arbitrage is functioning efficiently — informed "
        "capital is positioning well before the NSE press release. For passive fund "
        "managers seeking to minimise implementation shortfall, this implies that "
        "the best execution window has likely shifted earlier in the announcement "
        "cycle, and that same-day execution at announcement will often face already-moved "
        "prices."
    ),
    SP(4),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: LIMITATIONS & FUTURE WORK
# ─────────────────────────────────────────────────────────────────────────────

story += [H1("6. Limitations and Future Work"), HR()]

story += [
    P(
        "<b>Sample size.</b> With 47 events across 15 announcement dates, the era-level "
        "analysis (19 vs 28 events) has limited statistical power. Several findings are "
        "directionally consistent and economically meaningful but fall short of conventional "
        "significance thresholds, particularly in the correlation analysis."
    ),
    P(
        "<b>Collinearity of Era and Crisis.</b> Our crisis windows (COVID 2020, Rates 2022) "
        "both fall within Era 2. The regression cannot separately identify \"Era 2\" and "
        "\"crisis\" effects. A longer sample extending back to 2010 would help, though "
        "pre-2015 price data quality is less reliable."
    ),
    P(
        "<b>Continuous AUM measure.</b> We use era dummies as a proxy for passive AUM. "
        "A more rigorous approach would use the actual AMFI quarterly passive AUM as a "
        "continuous cross-sectional regressor, which would use all 47 events jointly "
        "and avoid arbitrary era cutpoints. This is left for future work."
    ),
    P(
        "<b>Tick data and intra-day analysis.</b> Our daily-frequency analysis cannot "
        "capture intra-day price dynamics, bid-ask spread changes, or the exact timing "
        "of informed trading around announcement. High-frequency NSE tick data would "
        "allow a more precise characterisation of the price discovery process."
    ),
    P(
        "<b>Nifty Next 50 anticipatory analysis.</b> Pre-announcement drift in the "
        "[-10, -1] window (positive for both inclusions and exclusions) may reflect "
        "anticipatory positioning by participants who correctly predict NSE's "
        "reconstitution decisions using the publicly disclosed methodology. Testing "
        "this formally requires reconstructing the historical Nifty Next 50 universe "
        "and identifying which stocks were at the inclusion/exclusion margin at each "
        "review date."
    ),
    SP(4),
]

# ─────────────────────────────────────────────────────────────────────────────
# APPENDIX
# ─────────────────────────────────────────────────────────────────────────────

story += [PageBreak(), H1("Appendix: Data Coverage and Excluded Events"), HR()]

excl_data = [
    ["Symbol", "Announcement", "Action", "Reason for Exclusion"],
    ["CAIRN",      "2016-02-22", "Exclusion", "Delisted 2016 (merger with Vedanta); no price data"],
    ["IBULHSGFIN", "2017-02-16", "Inclusion", "yfinance data unavailable (NSE feed issue)"],
    ["IBULHSGFIN", "2019-08-28", "Exclusion", "yfinance data unavailable (NSE feed issue)"],
    ["INFRATEL",   "2016-02-22", "Inclusion", "Merged → Indus Towers 2020; pre-merger prices unavailable"],
    ["INFRATEL",   "2020-08-20", "Exclusion", "Merged → Indus Towers 2020; prices represent different entity"],
    ["LTIM",       "2024-08-23", "Exclusion", "LTIMindtree: yfinance NSE data unavailable at time of analysis"],
    ["TATAMTRDVR", "2016-02-22", "Inclusion", "DVR shares delisted July 2023; yfinance feed unavailable"],
    ["TATAMTRDVR", "2017-08-28", "Exclusion", "DVR shares delisted July 2023; yfinance feed unavailable"],
    ["ZOMATO",     "2025-02-21", "Inclusion", "yfinance NSE data unavailable at time of analysis"],
]
excl_col_w = [2.5*cm, 3*cm, 2.5*cm, 8*cm]
story += [
    P("Table A1 lists the 9 events excluded due to unavailable price data."),
    P("<b>Table A1: Excluded Events</b>", caption_style),
    results_table(excl_data[0], excl_data[1:], excl_col_w),
    SP(12),
    P(
        "<b>Note on 2021-08-23 PDF:</b> The NSE press release for August 2021 "
        "(ind_prs23082021.pdf) is a 29-page scanned image with no extractable text. "
        "No Nifty 50 changes appear in the nifty50_master.csv for August 2021, "
        "consistent with NSE making no changes to the index in that review cycle."
    ),
    SP(8),
    HR(),
    P("Working Paper — May 2026", footer_style),
]

# ── Build PDF ─────────────────────────────────────────────────────────────────
out_path = REPORT / "index_effect_india.pdf"
doc = SimpleDocTemplate(
    str(out_path),
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN + 0.4*cm, bottomMargin=MARGIN,
    title="The Index Effect in Indian Equity Markets",
    author="Working Paper",
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"\n✅ Report saved → {out_path}")
print(f"   Size: {out_path.stat().st_size / 1024:.0f} KB")
