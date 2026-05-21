"""
01_build_symbol_map.py
======================
Creates data/symbol_map.csv with columns:
  stock_name, nse_symbol, status, notes

status values:
  active   — still listed, yfinance data available
  renamed  — listed under different ticker now (yfinance may still serve historical data)
  delisted — no longer listed; yfinance has data up to delisting date
  merged   — absorbed into another company; yfinance data exists under old ticker until merger
  manual   — needs manual data sourcing

After writing the static map, validates each symbol against yfinance and marks
confirmed/unconfirmed.
"""

import pandas as pd
import yfinance as yf
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# ── Static mapping (hand-curated) ────────────────────────────────────────────
# Covers all 99 stocks without symbols in the master + verify the 80 with symbols.
SYMBOL_MAP = [
    # Pre-2015 stocks without symbols
    ("ABB India Ltd.",                                "ABB",          "active",   ""),
    ("Andhra Valley Power Supply Co. Ltd.",           "ANDHRVAL",     "delisted", "Delisted ~2000; very old data unlikely available"),
    ("Apollo Tyres Ltd.",                             "APOLLOTYRE",   "active",   ""),
    ("Arvind Mills Ltd",                              "ARVIND",       "active",   "Renamed Arvind Ltd."),
    ("Ashok Leyland Ltd.",                            "ASHOKLEY",     "active",   ""),
    ("Asian Paints Ltd.",                             "ASIANPAINT",   "active",   ""),
    ("Axis Bank Ltd.",                                "AXISBANK",     "active",   ""),
    ("Bajaj Auto Ltd",                                "BAJAJ-AUTO",   "active",   ""),
    ("Bajaj Auto Ltd.",                               "BAJAJ-AUTO",   "active",   ""),
    ("Bank of India",                                 "BANKINDIA",    "active",   ""),
    ("Bharti Tele-Ventures Ltd.",                     "BHARTIARTL",   "renamed",  "Renamed Bharti Airtel Ltd."),
    ("Brooke Bond Lipton India Ltd.",                 "BBLIL",        "delisted", "Merged into HUL ~1996"),
    ("Castrol (India) Ltd.",                          "CASTROLIND",   "active",   ""),
    ("Chambal Fertilizers & Chemicals Ltd.",          "CHAMBLFERT",   "active",   ""),
    ("Cipla Ltd.",                                    "CIPLA",        "active",   ""),
    ("Coal India Ltd.",                               "COALINDIA",    "active",   "IPO 2010"),
    ("Colgate Palmolive (I) Ltd",                     "COLPAL",       "active",   ""),
    ("Dabur India Ltd.",                              "DABUR",        "active",   ""),
    ("Digital Equipment (india) Ltd.",                "DICEQ",        "delisted", "US subsidiary, delisted ~2001"),
    ("Digital Globalsoft Ltd.",                       "DIGITALGLOB",  "delisted", "Acquired by HP ~2003"),
    ("Dr. Reddy's Laboratories Ltd.",                 "DRREDDY",      "active",   ""),
    ("EIH Ltd.",                                      "EIHOTEL",      "active",   ""),
    ("East India Hotels Ltd",                         "EIHOTEL",      "active",   "Renamed EIH Ltd."),
    ("Essar Gujarat Ltd.",                            "ESSARGU",      "delisted", "Delisted; merged into Essar group entities"),
    ("Gas Authority of India Limited",                "GAIL",         "active",   ""),
    ("Glaxo (India) Ltd.",                            "GLAXO",        "renamed",  "Renamed GlaxoSmithKline Pharmaceuticals Ltd."),
    ("GlaxoSmithkline Consumer Healthcare Ltd.",      "GSKCONS",      "delisted", "Acquired by HUL 2020; delisted"),
    ("Glaxosmithkline Pharmaceuticals Ltd.",          "GLAXO",        "active",   ""),
    ("Great Eastern Shipping Company Limited.",       "GESHIP",       "active",   ""),
    ("HCL Infosystems Ltd.",                          "HCLINFOSYS",   "delisted", "Delisted ~2023"),
    ("HCL Technologies Ltd.",                         "HCLTECH",      "active",   ""),
    ("Hero Honda Motors Limited",                     "HEROMOTOCO",   "renamed",  "Renamed Hero MotoCorp Ltd. 2011"),
    ("ICICI Bank Ltd.",                               "ICICIBANK",    "active",   ""),
    ("ICICI Ltd.",                                    "ICICIBANK",    "merged",   "ICICI Ltd merged into ICICI Bank 2002; data pre-merger under ICICI"),
    ("IDFC Ltd",                                      "IDFC",         "active",   "Demerged IDFC FIRST Bank; holding co still listed"),
    ("Indian Aluminium Co. Ltd.",                     "INDAL",        "merged",   "Acquired by Hindalco; delisted ~2000"),
    ("Indian Hotels Co. Ltd.",                        "INDHOTEL",     "active",   ""),
    ("Indian Petrochemicals Corporation Ltd.",        "IPCL",         "merged",   "Merged into Reliance Industries 2007; delisted"),
    ("Indian Rayon & Industries Ltd.",                "INDRAYON",     "merged",   "Became Aditya Birla Nuvo, then Grasim; delisted"),
    ("Indo Gulf Corporation Ltd.",                    "INDOGULF",     "merged",   "Merged into Hindalco ~2004; delisted"),
    ("Industrial Development Bank of India Limited",  "IDBI",         "active",   "Converted to IDBI Bank Ltd."),
    ("Industrial Finance Corporation Of India Ltd.",  "IFCI",         "active",   ""),
    ("Infosys Technologies Limited",                  "INFY",         "renamed",  "Renamed Infosys Ltd."),
    ("Infrastructure Development Finance Company Lim...","IDFC",      "active",   "Same as IDFC Ltd"),
    ("JSW Steel Ltd.",                                "JSWSTEEL",     "active",   ""),
    ("Jaiprakash Associates Ltd.",                    "JPASSOCIAT",   "active",   ""),
    ("Jet Airways (India) Ltd.",                      "JETAIRWAYS",   "delisted", "Bankrupt; delisted 2019"),
    ("Kochi Refineries Ltd.",                         "KOCHI",        "merged",   "Merged into BPCL ~2006; delisted"),
    ("Kotak Mahindra Bank Ltd.",                      "KOTAKBANK",    "active",   ""),
    ("Larsen & Toubro Ltd.",                          "LT",           "active",   ""),
    ("Lupin Ltd.",                                    "LUPIN",        "active",   ""),
    ("Madras Refineries Ltd.",                        "MRL",          "renamed",  "Renamed Chennai Petroleum Corporation Ltd. (CHENNPETRO)"),
    ("Mahanagar Telephone Nigam Ltd.",                "MTNL",         "active",   ""),
    ("Mahindra & Mahindra Ltd.",                      "M&M",          "active",   ""),
    ("Mangalore Refinery & Petrochemicals Ltd.",      "MRPL",         "active",   ""),
    ("Maruti Udyog Limited",                          "MARUTI",       "renamed",  "Renamed Maruti Suzuki India Ltd."),
    ("NIIT Ltd.",                                     "NIIT",         "active",   ""),
    ("NTPC Ltd.",                                     "NTPC",         "active",   ""),
    ("Nagarjuna Fertilizers & Chemicals Ltd.",        "NAGARFERT",    "delisted", "Restructured; delisted ~2015"),
    ("National Aluminium Co. Ltd.",                   "NATIONALUM",   "active",   ""),
    ("Nestle India Limited",                          "NESTLEIND",    "active",   ""),
    ("Novartis India Ltd",                            "NOVARTIND",    "active",   ""),
    ("Novartis India Ltd.",                           "NOVARTIND",    "active",   ""),
    ("Oil & Natural Gas Corporation Ltd.",            "ONGC",         "active",   ""),
    ("Oriental Bank of Commerce",                     "ORIENTBANK",   "merged",   "Merged into PNB 2020; data available until merger"),
    ("Ponds (India) Ltd.",                            "PONDS",        "delisted", "Merged into HUL ~1998"),
    ("Power Grid Corporation of India Ltd.",          "POWERGRID",    "active",   ""),
    ("Procter & Gamble India Ltd.",                   "PGHH",         "active",   "P&G Hygiene & Health Care Ltd."),
    ("Ranbaxy Laboratories Ltd.",                     "RANBAXY",      "merged",   "Acquired by Sun Pharma; delisted 2015"),
    ("Reckitt & Colman India Ltd.",                   "RECKITTBENK",  "renamed",  "Renamed Reckitt Benckiser India Ltd."),
    ("Reckitt Benckiser (India) Ltd",                 "RECKITTBENK",  "active",   ""),
    ("Reliance Capital Ltd.",                         "RELCAPITAL",   "delisted", "Bankrupt; delisted 2022"),
    ("Reliance Communications Ltd.",                  "RCOM",         "delisted", "Bankrupt; delisted 2019"),
    ("Reliance Infrastructure Ltd.",                  "RELINFRA",     "active",   ""),
    ("Reliance Petroleum Ltd.",                       "RPL",          "merged",   "Merged into Reliance Industries 2009; delisted"),
    ("Reliance Power Ltd.",                           "RPOWER",       "active",   ""),
    ("SCICI Ltd.",                                    "SCICI",        "merged",   "Merged into ICICI Bank ~1996; very old data"),
    ("Satyam Computer Services Ltd.",                 "SATYAMCOMP",   "merged",   "Taken over by Tech Mahindra 2009; renamed TECHM"),
    ("Sesa Goa Limited",                              "SESAGOA",      "merged",   "Merged into Vedanta Ltd. 2013; now VEDL"),
    ("Shipping Corporation of India Ltd.",            "SCI",          "active",   ""),
    ("Siemens Ltd.",                                  "SIEMENS",      "active",   ""),
    ("Smithkline Beecham Consumer Healthcare Ltd.",   "GSKCONS",      "renamed",  "Renamed GlaxoSmithKline Consumer; then acquired by HUL"),
    ("Steel Authority of India Ltd.",                 "SAIL",         "active",   ""),
    ("Sterlite Industries (India) Ltd.",              "VEDL",         "merged",   "Merged into Vedanta Ltd. 2012; data under VEDL from 2013"),
    ("Sun Pharmaceutical Industries Ltd.",            "SUNPHARMA",    "active",   ""),
    ("Suzlon Energy Ltd.",                            "SUZLON",       "active",   ""),
    ("TVS Suzuki Ltd.",                               "TVSMOTOR",     "renamed",  "Renamed TVS Motor Company Ltd."),
    ("Tata Chemicals Ltd.",                           "TATACHEM",     "active",   ""),
    ("Tata Communications Ltd.",                      "TATACOMM",     "active",   "Formerly Videsh Sanchar Nigam Ltd."),
    ("Tata Consultancy Services Ltd.",                "TCS",          "active",   "IPO 2004"),
    ("Tata Tea Limited",                              "TATATEA",      "merged",   "Renamed Tata Global Beverages, now Tata Consumer Products (TATACONSUM)"),
    ("Tech Mahindra Ltd.",                            "TECHM",        "active",   ""),
    ("Thermax Ltd.",                                  "THERMAX",      "active",   ""),
    ("UltraTech Cement Ltd.",                         "ULTRACEMCO",   "active",   ""),
    ("Unitech Ltd.",                                  "UNITECH",      "active",   ""),
    ("United Spirits Ltd.",                           "MCDOWELL-N",   "active",   "Diageo subsidiary; listed as United Spirits"),
    ("Videsh Sanchar Nigam Ltd.",                     "VSNL",         "renamed",  "Renamed Tata Communications Ltd. (TATACOMM)"),
    ("Wipro Ltd.",                                    "WIPRO",        "active",   ""),
    ("Zee Telefilms Ltd",                             "ZEEL",         "renamed",  "Renamed Zee Entertainment Enterprises Ltd."),
]

# ── Build and save initial map ────────────────────────────────────────────────
sym_df = pd.DataFrame(SYMBOL_MAP, columns=["stock_name", "nse_symbol", "status", "notes"])
sym_df = sym_df.drop_duplicates("stock_name").reset_index(drop=True)

# ── Validate against yfinance ─────────────────────────────────────────────────
print("Validating symbols against yfinance...")
confirmed = []
for _, row in sym_df.iterrows():
    sym = row["nse_symbol"]
    try:
        hist = yf.Ticker(f"{sym}.NS").history(period="5d", auto_adjust=True)
        ok = len(hist) > 0
    except Exception:
        ok = False
    confirmed.append(ok)
    status = "✓" if ok else "✗"
    print(f"  {status} {sym:20s}  {row['stock_name'][:50]}")
    time.sleep(0.1)

sym_df["yfinance_ok"] = confirmed
out = DATA / "symbol_map.csv"
sym_df.to_csv(out, index=False)

ok_count = sum(confirmed)
print(f"\nSaved {len(sym_df)} symbols → {out}")
print(f"yfinance confirmed: {ok_count}/{len(sym_df)} ({ok_count/len(sym_df)*100:.0f}%)")
print(f"Not confirmed: {len(sym_df)-ok_count} (delisted / old tickers / needs manual data)")
