import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def score_headlines(df_news: pd.DataFrame) -> pd.DataFrame:
    df = df_news.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["compound"] = (
        df["headline"]
        .astype(str)
        .apply(lambda t: analyzer.polarity_scores(t)["compound"])
    )
    daily = (
        df.groupby("date")["compound"].agg(["mean", "median", "count"]).reset_index()
    )
    daily = daily.rename(
        columns={"mean": "sent_mean", "median": "sent_median", "count": "sent_count"}
    )
    return daily


def merge_sentiment(price_df: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    out["date"] = out.index.date
    sent = daily_sent.copy()
    sent["date"] = pd.to_datetime(sent["date"]).dt.date
    out = out.merge(sent, on="date", how="left").drop(columns=["date"])
    out[["sent_mean", "sent_median", "sent_count"]] = out[
        ["sent_mean", "sent_median", "sent_count"]
    ].fillna(0.0)
    return out
