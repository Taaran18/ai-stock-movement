import argparse, json, joblib, pandas as pd
from utils import load_config
from data_fetch import fetch_ohlcv
from features import add_price_features, build_feature_matrix

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["data"]["ticker"])
    args = ap.parse_args()

    model = joblib.load(f"{cfg['artifacts_dir']}/model.pkl")
    scaler = joblib.load(f"{cfg['artifacts_dir']}/scaler.pkl")
    with open(f"{cfg['artifacts_dir']}/feature_schema.json") as f:
        X_cols = json.load(f)

    df = fetch_ohlcv(args.ticker, cfg["data"]["period"], cfg["data"]["interval"])
    df_feat = add_price_features(df, cfg)
    X_latest = df_feat[X_cols].iloc[[-1]]  # last available day (for t)
    X_latest_scaled = scaler.transform(X_latest)

    pred = model.predict(X_latest_scaled)[0]
    proba = getattr(model, "predict_proba", lambda x: None)(X_latest_scaled)
    label_map = {0: "DOWN", 1: "UP", 2: "FLAT"}
    out = {
        "ticker": args.ticker,
        "date": df_feat.index[-1].strftime("%Y-%m-%d"),
        "prediction": int(pred),
        "label": label_map[int(pred)],
        "proba": (proba[0].tolist() if proba is not None else None),
    }
    print(json.dumps(out, indent=2))
