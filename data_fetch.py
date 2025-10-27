import pandas as pd
import yfinance as yf

_EXPECTED_FIELDS = {"open", "high", "low", "close", "adj close", "adj_close", "volume"}


def _normalize_cols_to_canon(df: pd.DataFrame) -> pd.DataFrame:
    def canon(c: str) -> str:
        k = str(c).strip().lower().replace("__", "_").replace("  ", " ")
        k = k.replace("_", " ").strip()
        if k == "adj close":
            nm = "Adj_Close"
        elif k == "open":
            nm = "Open"
        elif k == "high":
            nm = "High"
        elif k == "low":
            nm = "Low"
        elif k == "close":
            nm = "Close"
        elif k == "volume":
            nm = "Volume"
        else:
            nm = c
        return nm

    return df.rename(columns={c: canon(c) for c in df.columns})


def fetch_ohlcv(ticker: str, period="5y", interval="1d") -> pd.DataFrame:
    t_upper = ticker.split(".")[0].upper()

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        threads=False,
    )

    if df.empty:
        raise ValueError(
            f"yfinance returned empty data for {ticker} (period={period}, interval={interval})."
        )

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        unique_lvl0 = sorted({str(x).upper() for x in lvl0})

        if t_upper in unique_lvl0:
            try:
                df = df.xs(key=t_upper, axis=1, level=0, drop_level=True)
            except KeyError:
                try:
                    df = df.xs(key=ticker, axis=1, level=0, drop_level=True)
                except KeyError:
                    raise KeyError(
                        f"Ticker {ticker} not found in MultiIndex columns: {unique_lvl0}"
                    )
        else:
            if len(unique_lvl0) == 1:
                df.columns = df.columns.get_level_values(1)
            else:
                raise KeyError(
                    f"Could not resolve OHLCV columns. Level-0 values: {unique_lvl0}"
                )

    df = _normalize_cols_to_canon(df)

    if "Close" not in df.columns:
        if "Adj_Close" in df.columns:
            df["Close"] = df["Adj_Close"]
        else:
            raise KeyError(
                f"Expected 'Close' (or 'Adj Close') not found. Columns: {df.columns.tolist()}"
            )

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in keep:
        raise KeyError(
            f"'Close' still missing after normalization. Columns: {df.columns.tolist()}"
        )

    df = df[keep].dropna().sort_index()
    df.index.name = "Date"
    return df
