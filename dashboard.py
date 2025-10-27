import json, joblib, pandas as pd, numpy as np, streamlit as st, matplotlib.pyplot as plt
from utils import load_config
from data_fetch import fetch_ohlcv
from features import add_price_features, build_feature_matrix

st.set_page_config(page_title="AI Stock Movement Predictor", layout="wide")


@st.cache_data(show_spinner=False)
def get_data(ticker, period, interval):
    df = fetch_ohlcv(ticker, period, interval)
    return df


def plot_price(df_feat):
    fig, ax = plt.subplots()
    ax.plot(df_feat.index, df_feat["Close"], label="Close")
    for w in [5, 10, 20]:
        c = f"MA_{w}"
        if c in df_feat.columns:
            ax.plot(df_feat.index, df_feat[c], label=c, alpha=0.8)
    ax.set_title("Price & Moving Averages")
    ax.legend()
    st.pyplot(fig)


def plot_rsi(df_feat):
    if "RSI" not in df_feat.columns:
        return
    fig, ax = plt.subplots()
    ax.plot(df_feat.index, df_feat["RSI"], label="RSI")
    ax.axhline(70, linestyle="--")
    ax.axhline(30, linestyle="--")
    ax.set_title("RSI")
    st.pyplot(fig)


def plot_macd(df_feat):
    if "MACD" not in df_feat.columns:
        return
    fig, ax = plt.subplots()
    ax.plot(df_feat.index, df_feat["MACD"], label="MACD")
    ax.plot(df_feat.index, df_feat["MACD_signal"], label="Signal")
    ax.set_title("MACD")
    ax.legend()
    st.pyplot(fig)


cfg = load_config()
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value=cfg["data"]["ticker"])
period = st.sidebar.selectbox("History", ["1y", "2y", "5y", "10y", "max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)
run_btn = st.sidebar.button("Fetch & Predict")

col1, col2 = st.columns([2, 1])
with col1:
    st.title("AI Stock Movement Predictor")
    st.markdown("Predict next-day direction: **UP / DOWN / FLAT**")

if run_btn:
    df = get_data(ticker, period, interval)
    df_feat = add_price_features(df, cfg)

    # Load artifacts
    try:
        model = joblib.load(f"{cfg['artifacts_dir']}/model.pkl")
        scaler = joblib.load(f"{cfg['artifacts_dir']}/scaler.pkl")
        with open(f"{cfg['artifacts_dir']}/feature_schema.json") as f:
            X_cols = json.load(f)

        X_latest = df_feat[X_cols].iloc[[-1]]
        X_scaled = scaler.transform(X_latest)
        pred = model.predict(X_scaled)[0]
        proba = getattr(model, "predict_proba", lambda x: None)(X_scaled)
        label_map = {0: "DOWN", 1: "UP", 2: "FLAT"}
        label = label_map[int(pred)]
        conf = float(np.max(proba)) if proba is not None else None

        with col2:
            st.subheader("Prediction")
            st.metric(label="Next-Day Direction", value=label, delta=None)
            if conf is not None:
                st.progress(min(max(conf, 0), 1.0))
                st.caption(f"Model confidence: {conf:.2%}")

        with st.expander("Latest Features (last row)"):
            st.write(X_latest.T.rename(columns={X_latest.index[-1]: "value"}))

        plot_price(df_feat)
        plot_rsi(df_feat)
        plot_macd(df_feat)
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Train a model first (run `python train_model.py`)."
        )
else:
    st.info("Set a ticker and click **Fetch & Predict**.")

st.caption("Note: For research only. Not investment advice.")
