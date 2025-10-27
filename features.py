import pandas as pd, numpy as np, json
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def add_price_features(df, cfg):
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    for w in cfg["features"]["ma_windows"]:
        ma_col = f"MA_{w}"
        out[ma_col] = out["Close"].rolling(window=w).mean()
        out[f"{ma_col}_pct"] = out["Close"] / out[ma_col] - 1

    rsi = RSIIndicator(close=out["Close"], window=cfg["features"]["rsi_window"])
    out["RSI"] = rsi.rsi()

    macd = MACD(
        close=out["Close"],
        window_slow=cfg["features"]["macd_slow"],
        window_fast=cfg["features"]["macd_fast"],
        window_sign=cfg["features"]["macd_signal"],
    )
    out["MACD"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_diff"] = macd.macd_diff()

    bb = BollingerBands(
        close=out["Close"],
        window=cfg["features"]["bb_window"],
        window_dev=cfg["features"]["bb_std"],
    )
    out["BB_high"] = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_pct"] = (out["Close"] - out["BB_low"]) / (out["BB_high"] - out["BB_low"])

    for l in cfg["features"]["lags"]:
        out[f"Ret_lag_{l}"] = out["Return"].shift(l)

    out["Target_next_ret"] = out["Close"].pct_change().shift(-1)
    band = cfg["features"]["flat_threshold_bps"] / 10000.0

    def to_class(r):
        if pd.isna(r):
            return np.nan
        if abs(r) <= band:
            return 2
        return 1 if r > 0 else 0

    out["Target"] = out["Target_next_ret"].apply(to_class)

    out = out.dropna()
    return out


def build_feature_matrix(df):
    cols = [
        c
        for c in df.columns
        if c not in ["Target_next_ret", "Target", "Open", "High", "Low"]
    ]
    X_cols = [c for c in cols if c not in ["Close"]] + ["Close"]
    return X_cols


if __name__ == "__main__":
    pass
