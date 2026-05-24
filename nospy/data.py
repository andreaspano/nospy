import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    dfs = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        df_ticker = (
            yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
            )
            .reset_index()
        )

        # yfinance can return MultiIndex columns, especially with multiple tickers.
        if hasattr(df_ticker.columns, "get_level_values"):
            df_ticker.columns = df_ticker.columns.get_level_values(0)

        df_ticker.columns.name = None

        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df_ticker.columns]

        if missing_cols:
            raise ValueError(f"{ticker} is missing columns: {missing_cols}")

        df_ticker = (
            df_ticker[required_cols]
            .dropna()
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        df_ticker["unique_id"] = ticker
        dfs.append(df_ticker)

    if not dfs:
        raise ValueError("No ticker data was downloaded.")

    return pd.concat(dfs, ignore_index=True)


def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.copy()
    ts["ds"] = pd.to_datetime(ts["Date"])
    ts = ts.sort_values(["unique_id", "ds"])
    ts["y"] = ts.groupby("unique_id")["Close"].pct_change()
    ts = ts[["unique_id", "ds", "y"]]
    ts = ts.dropna().reset_index(drop=True)
    return ts
