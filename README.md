# The Index Effect in Indian Equity Markets (2015–2025)

### *Did the rise of passive investing kill the Nifty 50 index effect?*

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://indexeffect-prasanna.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance-0a66c2?style=for-the-badge)
![Data](https://img.shields.io/badge/Events-51%20Reconstitutions%20%7C%2044%20Stocks-2ea44f?style=for-the-badge)

---

## At a Glance

> **Passive AUM in India grew 26× — from ₹41,000 Cr to ₹11 lakh Cr — between 2015 and 2025.**
> This project asks a pointed empirical question: as index-tracking capital exploded, did front-runners arbitrage the Nifty 50 index effect out of existence?
>
> **Answer: Yes. Completely.**

| Metric | Era 1 (2015–17) | Era 2 (2018–25) |
|---|---|---|
| Inclusion CAR [0,+1] | **+2.14%** *(significant)* | **−0.27%** *(zero)* |
| Exclusion CAR | −0.82% | −0.82% *(persistent)* |
| VaR spike at events | +40–60% above baseline | +40–60% above baseline |
| AUM mechanism | — | −176 bp per 10× AUM increase (p=0.026) |

---

## What Makes This Project Technically Interesting

- **No off-the-shelf dataset** — NSE does not publish structured reconstitution data. Every event was scraped from raw PDF circulars using a custom parser (`nifty50_scraper.py`) that handles a silent format change NSE made around 2020.
- **Rigorous event study** — Market model with 260-day estimation windows, BMP standardised cross-sectional t-test (not the naive t-test that overstates significance under cross-sectional correlation).
- **Continuous AUM regressor** — Replaces discrete era dummies with log(PassiveAUM), allowing the mechanism to be quantified: every tenfold increase in passive AUM is associated with a 176 bp compression in inclusion returns.
- **Structural break detection** — Sup-Wald / Quandt-Andrews test + PELT change-point segmentation to find the break date from the data, not impose it.
- **Live interactive dashboard** — Six-page Streamlit app with live OLS coefficient updates as you drag a break-date slider.

---

## Key Findings

### Returns: The Effect Has Disappeared

![CAR trajectory across all events](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig1_car_trajectory.png)

*Cumulative Abnormal Returns (CAR) over the event window — [−10 to +40] trading days around announcement.*

![Era comparison — CAR distributions](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig2_car_era_comparison.png)

*Era 1 (2015–17) announcement-day CARs are clearly positive; Era 2 (2018–25) centers on zero.*

![CAR by era — inclusions vs exclusions](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig3_car_by_era.png)

*Inclusions fully compress. Exclusion penalty persists — asymmetry consistent with forced selling by passive funds.*

---

### Risk: VaR Spikes Are Structural, Not Gone

![VaR spike trajectory](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig4_var_trajectory.png)

*20-day rolling VaR (95th percentile), normalised to pre-event baseline. Spikes of +40–60% are visible at event windows for both inclusions and exclusions.*

![VaR — era comparison](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig5_var_era_comparison.png)

![VaR — crisis vs non-crisis](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig8_crisis_var_comparison.png)

*Crisis periods (COVID 2020, rate hike cycle 2022) amplify VaR spikes by an additional 0.6× multiplier.*

---

### Volume: Front-Running Is Already Priced In

![Abnormal volume at announcement](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig9_abnormal_volume.png)

*Inclusions trade at only **0.78×** average volume on announcement day — below baseline — consistent with the market having already priced the event before the official announcement.*

---

### Correlation Structure

![Correlation jump distribution](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig6_corr_jump_distribution.png)

![Correlation trajectory](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig7_corr_trajectory.png)

*60-day rolling Pearson correlation with ^NSEI. Newly included stocks show a structural jump in market beta at the effective date.*

---

### AUM Mechanism: The Quantified Cause

![CAR vs AUM scatter](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig14_aum_car.png)

![AUM timeline](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig15_aum_timeline.png)

*Continuous OLS regression: `CAR[0,+1] = β₀ + β₁·log(AUM) + β₂·Inclusion + β₃·log(AUM)×Inclusion + ε`*
*Coefficient on the interaction term: **−176 bp per 10× AUM increase** (p = 0.026).*

---

### Structural Break Analysis

![Sup-Wald break test](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig11_structural_break.png)

![PELT change-point segmentation](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig12_pelt_segmentation.png)

*Sup-F = 8.67 vs critical 8.85 — borderline. Maximum F-stat at **August 2024**, suggesting gradual compression rather than a single discrete break.*

![Era comparison summary](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig13_era_comparison.png)

---

### CAR vs VaR — Returns Disappear, Risk Doesn't

![CAR vs VaR scatter](https://raw.githubusercontent.com/prasanna1504/Index-Effect/main/results/figures/fig10_car_vs_var_scatter.png)

*The anomalous return disappears as passive AUM grows, but the risk spike does not — passive funds must still execute large trades regardless of whether the price has already moved.*

---

## Data Construction — Scraping NSE Circulars

There is no clean, downloadable dataset of historical Nifty 50 reconstitutions. NSE publishes each index change as a **PDF circular** on their website. Building the event dataset required scraping and parsing 10 years of these PDFs.

### The Problem
NSE releases index reconstitution circulars twice a year (February and August). Each circular covers **all NSE indices** in a single document — Nifty 50 is just one section buried among 30+ others:

```
Annexure I — Changes in NSE Indices
...
1) Nifty 50
   Effective from [date]:  INCLUDED: SYMBOL1   EXCLUDED: SYMBOL3
...
12) Nifty Midcap 150   ...
```

### The Silent Format Change
Around 2020, NSE changed its circular format — section numbering switched from numeric to alphabetic (`1) Nifty 50` → `a) Nifty 50`). A regex targeting only `\d+\) Nifty 50` **silently missed every circular from 2020 onward**, producing zero events for 5 years with no error raised.

```python
# Before — missed all circulars from ~2020 onwards
_NIFTY50_HDR = re.compile(r"^\(?\d+\)\s+Nifty\s+50\b", re.I)

# After — handles both numbered and lettered prefixes
_NIFTY50_HDR = re.compile(r"^(?:\(?\d+\)|[a-z]\))\s+Nifty\s+50\b", re.I)
```

### Result
- **14 announcement cycles** scraped (Feb/Aug each year, 2015–2025)
- **3 missing cycles** added manually after cross-checking NSE press releases
- **7 cycles** confirmed to have had no Nifty 50 changes
- **Final dataset: 51 events across 44 unique stocks**

---

## Methodology

### Event Study (Market Model)

```
Rᵢₜ = αᵢ + βᵢ · R_market,t + εᵢₜ
```

| Parameter | Value |
|---|---|
| Estimation window | [−260, −11] trading days |
| Event windows | [−10,−1] pre-drift · [0,+1] announcement · [+2,+5] · [+2,+20] · [+2,+40] post |
| Test statistic | BMP (Boehmer-Musumeci-Poulsen) standardised cross-sectional t-test |

> **Why BMP?** The naive t-test assumes independence across stocks. During index events, all affected stocks are exposed to the same announcement simultaneously — cross-sectional correlation inflates t-statistics. BMP corrects for this by standardising each abnormal return by its own prediction-error variance.

### Risk Analysis
- **VaR:** 20-day rolling historical 95th-percentile, normalised to [−30,−2] pre-event baseline
- **Correlation:** 60-day rolling Pearson with ^NSEI, jump measured at effective date
- **Abnormal volume:** Announcement-day volume ÷ mean over [−30,−2]

### Regime Analysis
- Crisis windows: COVID (Jan–Sep 2020), Rate hike cycle (Jan–Dec 2022)
- OLS: `peak_var_ratio ~ era2 + crisis + era2×crisis`

### Structural Break Detection
- **Sup-Wald / Quandt-Andrews test** (Andrews 1993) — data-driven break date search over trimmed interior
- **PELT** (ruptures, BIC penalty) — change-point segmentation

### AUM Mechanism
```
CAR[0,+1] = β₀ + β₁·log(AUM) + β₂·Inclusion + β₃·log(AUM)×Inclusion + ε
```
Replaces discrete era dummies with continuous passive AUM — uses all 51 events jointly and yields an interpretable, economically motivated coefficient.

---

## Dataset

| Source | Coverage |
|---|---|
| NSE PDF circulars (custom-scraped) | 51 reconstitution events, 2015–2025 |
| yfinance daily OHLCV | 44 stocks + ^NSEI benchmark, 2013–2026 |
| AMFI quarterly passive AUM | 21 observations, 2015–2025 |

Tickers with name changes resolved: `IBULHSGFIN → SAMMAANCAP.NS`, `ZOMATO → ETERNAL.NS`, `LTIM → LTM.BO`

---

## Tech Stack

| Category | Tools |
|---|---|
| Data & Analysis | `pandas` `numpy` `statsmodels` `scipy` |
| Visualisation | `plotly` `matplotlib` |
| Dashboard | `streamlit` |
| Price Data | `yfinance` |
| PDF Scraping | `pdfplumber` `re` |
| Change-point | `ruptures` |
| Reporting | `reportlab` |

---

## Project Structure

```
Index-Effect/
├── dashboard.py                    # Six-page Streamlit interactive dashboard
├── nifty50_scraper.py              # NSE circular PDF scraper
├── nifty50_changes.csv             # 51 scraped reconstitution events
├── requirements.txt
│
├── data/
│   ├── nifty50_master_clean.csv    # 214-row event master (with symbols)
│   ├── passive_aum.csv             # 21 quarterly passive AUM observations
│   └── prices/                     # Daily OHLCV — 44 stocks + ^NSEI
│
├── scripts/
│   ├── 00_fix_data.py              # Adds missing events, cleans master
│   ├── 01_fetch_prices.py          # yfinance bulk download
│   ├── 01b_fix_missing_prices.py   # Retry failed / renamed tickers
│   ├── 02_event_study.py           # Market model, CARs, BMP test
│   ├── 03_var_analysis.py          # Rolling VaR spike analysis
│   ├── 04_correlation_analysis.py  # Pre/post correlation jumps
│   ├── 05_regime_analysis.py       # Crisis vs calm, abnormal volume
│   ├── 06_generate_report.py       # PDF working paper (reportlab)
│   ├── 07_structural_break.py      # Sup-Wald + PELT break detection
│   └── 08_aum_regressor.py         # Passive AUM continuous regressor
│
└── results/
    ├── event_study_cars.csv
    ├── var_spikes.csv
    ├── correlation_jumps.csv
    ├── regime_analysis.csv
    ├── aum_regression.csv
    └── figures/                    # 15 publication-quality charts
```

---

## Dashboard

Six interactive pages — [live here](https://indexeffect-prasanna.streamlit.app/):

| Page | What it shows |
|---|---|
| **Overview** | KPI cards, event timeline, AUM growth chart |
| **Returns Explorer** | CAR distributions, time series, era comparison — any window, any filter |
| **Risk Explorer** | VaR spikes and correlation jumps by era and market regime |
| **Structural Break Lab** | Drag a break-date slider — F-stat updates live |
| **AUM Regressor** | CAR vs AUM scatter with live OLS coefficients |
| **Event Data** | Full 51-event table, filterable, downloadable |

---

## Running Locally

```bash
git clone https://github.com/prasanna1504/Index-Effect.git
cd Index-Effect
pip install -r requirements.txt
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

## Era Definitions

| Era | Period | Passive AUM Range | Index Effect |
|---|---|---|---|
| **Era 1** | 2015–2017 | ₹0.4L Cr → ₹1.2L Cr | Strong (+2.14%***) |
| **Era 2** | 2018–2025 | ₹1.2L Cr → ₹11L Cr | Eliminated (−0.27%) |

The 2018 cutpoint is economically motivated (passive AUM crossing ₹1L Cr). The Sup-Wald test finds a borderline F-statistic (8.67 vs 8.85 critical), with max F-stat at **August 2024** — consistent with gradual compression rather than a discrete structural break.

---

## Limitations

- Pre-2015 data excluded (less reliable manual sourcing)
- 6 tickers partially unavailable via yfinance due to name changes or delistings
- Passive AUM data uses AMFI-based estimates, not exact figures
- Crisis windows and Era 2 are collinear, limiting regime regression interpretation
- N = 51 total events — small sample for high-dimensional interaction models
