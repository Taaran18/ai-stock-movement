import argparse, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import load_config, ensure_dir, save_json, utcnow_str
from features import add_price_features, build_feature_matrix
from data_fetch import fetch_ohlcv


def time_split(df, test_size_days):
    cutoff = df.index.max() - pd.tseries.frequencies.to_offset(f"{test_size_days}D")
    train = df[df.index <= cutoff]
    test = df[df.index > cutoff]
    return train, test


def train_models(X_train, y_train, cfg):
    models = {}
    if "logreg" in cfg["training"]["models"]:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
        lr.fit(X_train, y_train)
        models["logreg"] = lr
    if "rf" in cfg["training"]["models"]:
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=cfg["training"]["random_state"],
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        models["rf"] = rf
    return models


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["data"]["ticker"])
    ap.add_argument("--use_cached_csv", action="store_true")
    args = ap.parse_args()

    ensure_dir(cfg["artifacts_dir"])
    if args.use_cached_csv:
        df = pd.read_csv(
            cfg["data"]["csv_path"], parse_dates=["Date"], index_col="Date"
        )
    else:
        df = fetch_ohlcv(args.ticker, cfg["data"]["period"], cfg["data"]["interval"])
        df.to_csv(cfg["data"]["csv_path"])

    df_feat = add_price_features(df, cfg)
    X_cols = build_feature_matrix(df_feat)
    y = df_feat["Target"].astype(int)
    X = df_feat[X_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_df, test_df = time_split(df_feat, cfg["training"]["test_size_days"])
    X_train = scaler.transform(train_df[X_cols])
    y_train = train_df["Target"].astype(int).values
    X_test = scaler.transform(test_df[X_cols])
    y_test = test_df["Target"].astype(int).values

    models = train_models(X_train, y_train, cfg)
    results = {}
    best_name, best_score = None, -1
    for name, mdl in models.items():
        y_pred = mdl.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        rpt = classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["DOWN(0)", "UP(1)", "FLAT(2)"],
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = rpt["macro avg"]["f1-score"]
        results[name] = {
            "macro_f1": macro_f1,
            "report": rpt,
            "confusion_matrix": cm.tolist(),
        }
        if macro_f1 > best_score:
            best_name, best_score = name, macro_f1

    joblib.dump(models[best_name], f"{cfg['artifacts_dir']}/model.pkl")
    joblib.dump(scaler, f"{cfg['artifacts_dir']}/scaler.pkl")
    meta = {
        "ticker": args.ticker,
        "timestamp_utc": utcnow_str(),
        "features": X_cols,
        "best_model": best_name,
        "results": results,
    }
    save_json(meta, f"{cfg['artifacts_dir']}/meta.json")
    with open(f"{cfg['artifacts_dir']}/feature_schema.json", "w") as f:
        json.dump(X_cols, f, indent=2)

    print(
        f"Saved model: {best_name} (macro F1={best_score:.3f}) to artifacts/model.pkl"
    )
