# The Index Effect in Indian Equity Markets (2015–2025)

**Does the Nifty 50 Index Effect still exist — and did the rise of passive investing kill it?**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://indexeffect-prasanna.streamlit.app/)

---

## What This Project Does

This is an end-to-end quantitative research project studying how Nifty 50 index reconstitutions affect stock returns and risk, and how that effect has changed as India's passive AUM grew from ₹41,000 Cr (2015) to ₹11 lakh Cr (2025) — a **26× increase**.

The research is split into two acts:

**Act 1 — Returns:** Do stocks added to/removed from Nifty 50 earn abnormal returns around the announcement? Has that effect compressed over time?

**Act 2 — Risk:** Do these events cause VaR spikes, correlation structure changes, and volume abnormalities? Do they get worse during market crises?

---

## Key Findings

| Finding | Result |
|---|---|
| Era 1 (2015–17) Inclusion CAR | **+2.14%*** on announcement day |
| Era 2 (2018–25) Inclusion CAR | **–0.27%** (statistically zero) |
| Exclusion CAR | **–0.82%** (persistent across both eras) |
| AUM mechanism | 10× AUM increase → **–176 bp** in inclusion CAR (p=0.026) |
| Structural break | Sup-F = 8.67 vs critical 8.85 — borderline; max F-stat at Aug 2024 |
| VaR spike around events | **+40–60% above baseline** (both inclusions and exclusions) |
| Crisis amplification | Adds **+0.6×** to VaR spike for inclusions |
| Abnormal volume | Inclusions trade at **0.78×** baseline (pre-announcement pricing-in) |

---

## Data Collection — Scraping NSE Circulars

There is no clean, downloadable dataset of historical Nifty 50 reconstitutions. NSE publishes each index change as a **PDF circular** on their website. Building the event dataset required scraping and parsing 10 years of these PDFs.

### The problem
NSE releases index reconstitution circulars twice a year (typically February and August). Each circular covers **all NSE indices** in a single document — Nifty 50 is just one section buried among 30+ others. The raw text looks like:

```
Annexure I
Changes in NSE Indices
...
1) Nifty 50
   Following changes are being made effective from [date]:
   INCLUDED:  SYMBOL1, SYMBOL2
   EXCLUDED:  SYMBOL3
...
12) Nifty Midcap 150
   ...
```

### What the scraper does (`nifty50_scraper.py`)
1. **Downloads PDF circulars** from NSE's index announcements archive (2015–2025)
2. **Extracts raw text** using `pdfplumber`
3. **Locates the Nifty 50 section** using a regex header match, then reads until the next index section begins
4. **Parses inclusions and exclusions** — company names are mapped to NSE ticker symbols

### Key engineering challenge — format change
Around 2020, NSE changed its circular format. The section numbering switched from:
```
1) Nifty 50        →    a) Nifty 50
2) Nifty Next 50   →    b) Nifty Next 50
```
A regex targeting only `1) Nifty 50` silently missed every circular from 2020 onward, producing **zero events for 5 years**. The fix required updating both the section header pattern and the section boundary detector to handle both numbered (`\d+\)`) and lettered (`[a-z]\)`) prefixes:

```python
# Before (missed ~2020 onwards)
_NIFTY50_HDR = re.compile(r"^\(?\d+\)\s+Nifty\s+50\b", re.I)

# After (handles both formats)
_NIFTY50_HDR = re.compile(r"^(?:\(?\d+\)|[a-z]\))\s+Nifty\s+50\b", re.I)
```

### Result
- **14 announcement cycles** scraped (Feb/Aug each year, 2015–2025)
- **3 missing cycles** identified and added manually (2020-Aug, 2024-Aug, 2025-Feb) after cross-checking against NSE press releases
- **7 cycles confirmed** to have had no Nifty 50 changes (verified from PDFs)
- Final dataset: **51 events** across 44 unique stocks

---

## Dataset

- **Events:** 51 Nifty 50 reconstitution events (2015–2025), scraped from NSE circulars
- **Symbols:** 44 unique stocks + ^NSEI benchmark
- **Prices:** Daily OHLCV from yfinance (2013–2026), auto-adjusted for splits/dividends
- **Passive AUM:** 21 quarterly observations from AMFI estimates (2015–2025)
- **Missing tickers resolved:** IBULHSGFIN → `SAMMAANCAP.NS`, ZOMATO → `ETERNAL.NS`, LTIM → `LTM.BO` (all name changes)

---

## Methodology

### Event Study (Market Model)
```
Rᵢₜ = αᵢ + βᵢ · R_market,t + εᵢₜ
```
- **Estimation window:** [–260, –11] trading days before announcement
- **Event windows:** [–10,–1] pre-drift, [0,+1] announcement, [+2,+5], [+2,+20], [+2,+40] post
- **Test statistic:** BMP (Boehmer-Musumeci-Poulsen) standardised cross-sectional t-test

### Risk Analysis
- **VaR:** 20-day rolling historical 95th-percentile VaR, normalised to [–30,–2] pre-event baseline
- **Correlation:** 60-day rolling Pearson correlation with ^NSEI, jump measured at effective date
- **Abnormal volume:** Announcement-day volume / mean volume over [–30,–2]

### Regime Analysis
- Crisis windows: COVID (Jan–Sep 2020), Rate hike cycle (Jan–Dec 2022)
- OLS: `peak_var_ratio ~ era2 + crisis + era2×crisis`

### Structural Break Detection
- **Sup-Wald / Quandt-Andrews test** (Andrews 1993) — data-driven break date search
- **PELT** (ruptures library, BIC penalty) — change-point segmentation

### AUM Mechanism (Continuous Regressor)
```
CAR[0,+1] = β₀ + β₁·log(AUM) + β₂·Inclusion + β₃·log(AUM)×Inclusion + ε
```
Replaces discrete era dummies with continuous passive AUM — methodologically superior as it uses all 51 events jointly and gives an interpretable coefficient.

---

## Project Structure

```
Index/
├── dashboard.py                  # Streamlit interactive dashboard
├── requirements.txt
├── nifty50_scraper.py            # NSE circular PDF scraper
├── nifty50_changes.csv           # Scraped reconstitution events
│
├── data/
│   ├── nifty50_master_clean.csv  # 214-row event master (with symbols)
│   ├── passive_aum.csv           # Quarterly passive AUM estimates
│   └── prices/                   # Daily OHLCV for 44 symbols + ^NSEI
│
├── scripts/
│   ├── 00_fix_data.py            # Adds missing events, cleans master
│   ├── 01_fetch_prices.py        # yfinance bulk download
│   ├── 01b_fix_missing_prices.py # Retry failed tickers
│   ├── 02_event_study.py         # Market model, CARs, BMP test
│   ├── 03_var_analysis.py        # Rolling VaR spike analysis
│   ├── 04_correlation_analysis.py# Pre/post correlation jumps
│   ├── 05_regime_analysis.py     # Crisis vs calm, abnormal volume
│   ├── 06_generate_report.py     # PDF working paper (reportlab)
│   ├── 07_structural_break.py    # Sup-Wald + PELT break detection
│   └── 08_aum_regressor.py       # Passive AUM continuous regressor
│
└── results/
    ├── event_study_cars.csv
    ├── var_spikes.csv
    ├── correlation_jumps.csv
    ├── regime_analysis.csv
    ├── aum_regression.csv
    └── figures/                  # 15 publication-quality charts
```

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/prasanna1504/Index-Effect.git
cd Index-Effect

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run dashboard.py
```

To re-run the full analysis pipeline from scratch:
```bash
python scripts/00_fix_data.py
python scripts/01_fetch_prices.py
python scripts/02_event_study.py
python scripts/03_var_analysis.py
python scripts/04_correlation_analysis.py
python scripts/05_regime_analysis.py
python scripts/07_structural_break.py
python scripts/08_aum_regressor.py
```

---

## Dashboard Pages

| Page | What it shows |
|---|---|
| **Overview** | KPI summary, event timeline, AUM growth chart |
| **Returns Explorer** | CAR distributions, time series, era comparison — any window, any filter |
| **Risk Explorer** | VaR spikes and correlation jumps by era and market regime |
| **Structural Break Lab** | Drag a break date slider — F-stat updates live |
| **AUM Regressor** | CAR vs AUM scatter with live OLS coefficients |
| **Event Data** | Full 51-event table, filterable, downloadable |

---

## Era Definitions

| Era | Period | Passive AUM Range | Index Effect |
|---|---|---|---|
| **Era 1** | 2015–2017 | ₹0.4L Cr → ₹1.2L Cr | Strong (+2.1%***) |
| **Era 2** | 2018–2025 | ₹1.2L Cr → ₹11L Cr | Compressed (–0.27%) |

The 2018 cutpoint is economically motivated (passive AUM crossed ₹1L Cr). The Sup-Wald structural break test finds a borderline F-statistic (8.67 vs 8.85 critical), with the maximum F-stat at **August 2024** — suggesting the effect compressed gradually as AUM grew, rather than at a single break.

---

## Limitations

- Pre-2015 data excluded (less reliable manual sourcing)
- 6 tickers partially unavailable via yfinance (name changes/delistings)
- Passive AUM data is AMFI-based estimates, not exact monthly figures
- Crisis and Era 2 are collinear (GFC predates the sample), limiting regime regression interpretation
- N=51 total events — small sample for high-dimensional interaction models

---

## Tech Stack

`Python` · `pandas` · `numpy` · `statsmodels` · `scipy` · `yfinance` · `plotly` · `streamlit` · `ruptures` · `reportlab`
