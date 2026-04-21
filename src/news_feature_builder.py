# src/news_feature_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np


@dataclass
class NewsFeatureBuilder:
    train_universe: List[str]

    def prepare_raw_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        df = news_df.copy()

        required_cols = ["Date", "Ticker", "headline"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required news columns: {missing}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["headline"] = df["headline"].fillna("").astype(str)

        if "summary" not in df.columns:
            df["summary"] = ""
        df["summary"] = df["summary"].fillna("").astype(str)

        if "source" not in df.columns:
            df["source"] = "unknown"
        df["source"] = df["source"].fillna("unknown").astype(str)

        optional_numeric = [
            "headline_sentiment",
            "summary_sentiment",
            "combined_sentiment",
            "sentiment_neg",
            "sentiment_neu",
            "sentiment_pos",
        ]
        for col in optional_numeric:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df = df[df["Ticker"].isin(self.train_universe)].copy()
        df = df.dropna(subset=["Date"]).reset_index(drop=True)

        df["headline_len"] = df["headline"].str.len()
        df["summary_len"] = df["summary"].str.len()

        return df

    def build_daily_news_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_raw_news(news_df)

        daily = (
            df.groupby(["Date", "Ticker"], as_index=False)
            .agg(
                news_count_1d=("headline", "count"),
                source_count_1d=("source", "nunique"),
                headline_length_mean_1d=("headline_len", "mean"),
                summary_length_mean_1d=("summary_len", "mean"),
                headline_sentiment_mean_1d=("headline_sentiment", "mean"),
                summary_sentiment_mean_1d=("summary_sentiment", "mean"),
                combined_sentiment_mean_1d=("combined_sentiment", "mean"),
                sentiment_neg_mean_1d=("sentiment_neg", "mean"),
                sentiment_neu_mean_1d=("sentiment_neu", "mean"),
                sentiment_pos_mean_1d=("sentiment_pos", "mean"),
            )
        )

        all_frames = []

        for ticker in self.train_universe:
            sub = daily[daily["Ticker"] == ticker].copy()
            sub = sub.sort_values("Date").reset_index(drop=True)

            if len(sub) == 0:
                continue

            sub["news_count_3d"] = sub["news_count_1d"].rolling(3).sum()
            sub["news_count_5d"] = sub["news_count_1d"].rolling(5).sum()

            sub["source_count_5d"] = sub["source_count_1d"].rolling(5).mean()

            sub["headline_sentiment_mean_3d"] = sub["headline_sentiment_mean_1d"].rolling(3).mean()
            sub["headline_sentiment_mean_5d"] = sub["headline_sentiment_mean_1d"].rolling(5).mean()

            sub["summary_sentiment_mean_3d"] = sub["summary_sentiment_mean_1d"].rolling(3).mean()
            sub["summary_sentiment_mean_5d"] = sub["summary_sentiment_mean_1d"].rolling(5).mean()

            sub["combined_sentiment_mean_3d"] = sub["combined_sentiment_mean_1d"].rolling(3).mean()
            sub["combined_sentiment_mean_5d"] = sub["combined_sentiment_mean_1d"].rolling(5).mean()

            sub["abnormal_news_count_20d"] = (
                sub["news_count_1d"] / (sub["news_count_1d"].rolling(20).mean() + 1e-9)
            )

            sub["sentiment_shock_5d"] = (
                sub["combined_sentiment_mean_1d"] - sub["combined_sentiment_mean_5d"]
            )

            all_frames.append(sub)

        if len(all_frames) == 0:
            return pd.DataFrame(columns=["Date", "Ticker"])

        out = pd.concat(all_frames, axis=0, ignore_index=True)
        out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)

        return out