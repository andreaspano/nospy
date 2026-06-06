"""
Data downloading and preparation utilities.
"""

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor


def _download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    ).reset_index()

    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker} is missing columns: {missing_cols}")

    df = (
        df[required_cols]
        .dropna()
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df["unique_id"] = ticker
    return df


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    print(f"Downloading {len(tickers)} tickers concurrently...")

    with ThreadPoolExecutor(max_workers=len(tickers)) as executor:
        dfs = list(executor.map(
            lambda t: _download_single_ticker(t, start_date, end_date, interval, auto_adjust),
            tickers,
        ))

    if not dfs:
        raise ValueError("No ticker data was downloaded.")

    return pd.concat(dfs, ignore_index=True)


def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.copy()
    ts["ds"] = pd.to_datetime(ts["Date"])
    ts["y"] = ts["Close"]
    ts = ts.sort_values(["unique_id", "ds"])
    ts = ts[["unique_id", "ds", "y"]]
    ts = ts.dropna().reset_index(drop=True)

    # Add TOTAL as an independently forecasted aggregate series.
    # This makes hierarchical reconciliation non-trivial: the model will
    # learn TOTAL's pattern directly, so its forecast may differ from the
    # sum of individual bottom-level forecasts before reconciliation.
    total = (
        ts.groupby("ds", as_index=False)["y"]
        .sum()
        .assign(unique_id="TOTAL")
    )
    ts = (
        pd.concat([ts, total[["unique_id", "ds", "y"]]], ignore_index=True)
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )

    return ts
