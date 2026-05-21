"""
Nifty 50 Constituent Change Scraper
=====================================
Downloads PDFs from NSE press releases (Feb & Aug announcements)
and extracts Nifty 50 additions/deletions.

Requirements:
    pip install requests pdfplumber pandas openpyxl

Usage:
    python nifty50_scraper.py

Output:
    nifty50_changes.csv  — one row per stock change with columns:
        announcement_date, effective_date, stock_name, action, source_pdf
"""

import re
import time
import requests
import pdfplumber
import pandas as pd
from pathlib import Path
from io import BytesIO

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "nifty50_feb_aug_releases.xlsx"
OUTPUT_CSV  = "nifty50_changes.csv"
PDF_DIR     = Path("pdfs")
PDF_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def derive_effective_date(announcement_date: pd.Timestamp) -> str:
    """
    Nifty 50 changes take effect on the last Friday of March (Feb announcement)
    or last Friday of September (Aug announcement).
    """
    month = announcement_date.month
    year  = announcement_date.year

    if month == 2:
        target_month, target_year = 3, year
    elif month == 8:
        target_month, target_year = 9, year
    else:
        return "unknown"

    last_day = pd.Timestamp(year=target_year, month=target_month, day=1) \
               + pd.offsets.MonthEnd(0)
    while last_day.weekday() != 4:  # 4 = Friday
        last_day -= pd.Timedelta(days=1)
    return last_day.strftime("%Y-%m-%d")


def download_pdf(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None


def extract_text(pdf_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  ✗ PDF parse error: {e}")
    return text


# Matches the Nifty 50 section header in all known NSE circular formats:
#   "(1) CNX Nifty Index"  /  "1) NIFTY 50"  /  "1) Nifty 50 Index"
#   "a) Nifty 50"  /  "a) NIFTY 50"  (sub-lettered format used from ~2020)
_NIFTY50_HDR = re.compile(
    r"^(?:\(?\d+\)|[a-z]\))\s+"
    r"(?:S&P\s+CNX\s+Nifty|CNX\s+Nifty|NIFTY\s*50|Nifty\s+50)"
    r"(?:\s+Index)?\s*$",
    re.IGNORECASE,
)
# Any numbered/lettered section header — used to detect the end of the Nifty 50 section.
_ANY_SECTION_HDR = re.compile(r"^(?:\(?\d+\)|[a-z]\))\s+\S")

# Action trigger lines
_EXCLUDED_PAT = re.compile(
    r"following\s+(?:compan(?:y|ies)|scrips?)\s+(?:is|are)\s+being\s+excluded",
    re.IGNORECASE,
)
_INCLUDED_PAT = re.compile(
    r"following\s+(?:compan(?:y|ies)|scrips?)\s+(?:is|are)\s+being\s+included",
    re.IGNORECASE,
)
_NO_CHANGES_PAT = re.compile(
    r"no\s+changes\s+are\s+being\s+made\s+in\s+nifty\s*50",
    re.IGNORECASE,
)
# Table header lines to skip (including split "Sr." / "No." from older PDFs)
_TABLE_HDR_PAT = re.compile(
    r"^(?:sr\.?\s*(?:no\.?)?|company\s+name|scrip\s+name|symbol)\s*$",
    re.IGNORECASE,
)
# Numbered table row: "1 Hero MotoCorp Ltd. HEROMOTOCO"
_TABLE_ROW_PAT = re.compile(r"^\d+\s+(.+)$")
# NSE trading symbol: all uppercase, 2–20 chars, may contain digits or &
_SYMBOL_PAT = re.compile(r"^[A-Z][A-Z0-9&\-]{1,19}$")


def _parse_row(raw: str) -> tuple[str, str] | tuple[None, None]:
    """
    Split a raw row value (after stripping the leading row number) into
    (company_name, symbol).  The symbol is always the last all-uppercase token.
    """
    tokens = raw.rstrip("* ").split()
    if not tokens:
        return None, None
    symbol = tokens[-1]
    if not _SYMBOL_PAT.match(symbol):
        return None, None
    company = " ".join(tokens[:-1]).rstrip("*").strip()
    if len(company) < 3:
        return None, None
    return company, symbol


def extract_changes(text: str, announcement_date: str, effective_date: str,
                    pdf_url: str) -> list[dict]:
    """
    Walk the PDF text line by line.

    State machine:
      • Enter Nifty 50 section when the section header is found.
      • Flip current_action to Exclusion/Inclusion on trigger phrases.
      • Parse numbered table rows while inside the section.
      • Exit on: "No changes" sentence, or the next numbered section header.
    """
    records       = []
    in_nifty50    = False
    current_action = None

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for line in lines:
        # ── Section entry ────────────────────────────────────────────────────
        if _NIFTY50_HDR.match(line):
            in_nifty50     = True
            current_action = None
            continue

        if not in_nifty50:
            continue

        # ── Section exit ─────────────────────────────────────────────────────
        if _NO_CHANGES_PAT.search(line):
            break

        # Next numbered section (uses ")" not ".") → we've left Nifty 50
        if _ANY_SECTION_HDR.match(line) and not _NIFTY50_HDR.match(line):
            break

        # ── Action trigger ───────────────────────────────────────────────────
        if _EXCLUDED_PAT.search(line):
            current_action = "Exclusion"
            continue

        if _INCLUDED_PAT.search(line):
            current_action = "Inclusion"
            continue

        if current_action is None:
            continue

        # ── Table header (skip) ──────────────────────────────────────────────
        if _TABLE_HDR_PAT.match(line):
            continue

        # ── Table data row ───────────────────────────────────────────────────
        m = _TABLE_ROW_PAT.match(line)
        if m:
            company, symbol = _parse_row(m.group(1))
            if company and symbol:
                records.append({
                    "announcement_date": announcement_date,
                    "effective_date":    effective_date,
                    "stock_name":        company,
                    "symbol":            symbol,
                    "action":            current_action,
                    "source_pdf":        pdf_url,
                })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Support both .xlsx and .csv input
    if INPUT_FILE.endswith(".xlsx"):
        releases = pd.read_excel(INPUT_FILE)
    else:
        releases = pd.read_csv(INPUT_FILE)

    # Normalise column names to uppercase for robustness
    releases.columns = [c.strip().upper() for c in releases.columns]

    date_col     = next(c for c in releases.columns if "DATE" in c)
    download_col = next(c for c in releases.columns if "DOWNLOAD" in c or "URL" in c or "LINK" in c)
    subject_col  = next((c for c in releases.columns if "SUBJECT" in c or "TITLE" in c), None)

    releases["DATE_parsed"] = pd.to_datetime(releases[date_col], dayfirst=True, errors="coerce")

    all_records = []

    for _, row in releases.iterrows():
        ann_date = row["DATE_parsed"]
        if pd.isna(ann_date):
            continue

        ann_str = ann_date.strftime("%Y-%m-%d")
        eff_str = derive_effective_date(ann_date)
        url     = str(row[download_col]).strip()
        subject = str(row[subject_col]).strip() if subject_col else url

        print(f"\n[{ann_str}] {subject}")
        print(f"  URL: {url}")

        pdf_filename = PDF_DIR / f"{ann_str}_{Path(url).name}"

        if pdf_filename.exists():
            print("  ✓ Cached — reading from disk")
            pdf_bytes = pdf_filename.read_bytes()
        else:
            pdf_bytes = download_pdf(url)
            if pdf_bytes is None:
                continue
            pdf_filename.write_bytes(pdf_bytes)
            print(f"  ✓ Downloaded ({len(pdf_bytes):,} bytes)")
            time.sleep(1.5)

        text = extract_text(pdf_bytes)
        records = extract_changes(text, ann_str, eff_str, url)

        if records:
            print(f"  ✓ Found {len(records)} change(s):")
            for r in records:
                print(f"      [{r['action']:9s}] {r['symbol']:20s} {r['stock_name']}")
            all_records.extend(records)
        else:
            print("  — No Nifty 50 changes (or no Nifty 50 section found)")
            print(f"    PDF saved to: {pdf_filename}")

    if all_records:
        out = pd.DataFrame(all_records).drop_duplicates()
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Done. {len(out)} changes saved to {OUTPUT_CSV}")
        print(out.to_string())
    else:
        print("\n⚠ No records extracted. Check the PDFs manually in the pdfs/ folder.")


if __name__ == "__main__":
    main()
