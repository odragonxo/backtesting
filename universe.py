"""S&P 500 constituents (fetched from Wikipedia, with a small built-in fallback)."""
import io

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Used only if the live list cannot be fetched (NOT the full index).
FALLBACK = (
    "AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B JPM LLY V UNH XOM MA COST HD PG JNJ ABBV NFLX BAC "
    "WMT CRM ORCL CVX KO MRK AMD CSCO PEP ADBE TMO LIN MCD ACN ABT DIS WFC GE INTU QCOM TXN AMAT"
).split()


def parse_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Wikipedia table -> DataFrame[ticker, name, sector]. Yahoo uses '-' instead of '.' (BRK.B -> BRK-B)."""
    out = pd.DataFrame({
        "ticker": tbl["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
        "name": tbl["Security"].astype(str),
        "sector": tbl["GICS Sector"].astype(str),
    })
    return out.drop_duplicates("ticker").reset_index(drop=True)


def fetch_sp500() -> pd.DataFrame:
    """Raises on network/parse failure so the caller can decide what to do."""
    import requests

    r = requests.get(WIKI_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0 (backtest-app)"})
    r.raise_for_status()
    df = parse_table(pd.read_html(io.StringIO(r.text))[0])
    if len(df) < 400:  # sanity check: page layout changed?
        raise ValueError(f"unexpected S&P 500 list size: {len(df)}")
    return df


def fallback_df() -> pd.DataFrame:
    return pd.DataFrame({"ticker": FALLBACK, "name": "", "sector": ""})
