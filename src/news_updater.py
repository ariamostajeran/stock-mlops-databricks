# src/news_updater.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import List
import pandas as pd
import requests


@dataclass
class NewsUpdater:
    api_key: str
    train_universe: List[str]

    def _standardize_news_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=[
                "published_at", "Date", "Ticker", "headline", "summary", "source", "url",
                "headline_sentiment", "summary_sentiment", "combined_sentiment",
                "sentiment_neg", "sentiment_neu", "sentiment_pos"
            ])

        out = df.copy()

        # force non-categorical stable dtypes
        str_cols = ["Ticker", "headline", "summary", "source", "url"]
        for col in str_cols:
            if col not in out.columns:
                out[col] = ""
            out[col] = out[col].astype("string").fillna("").astype(str)

        date_cols = ["published_at", "Date"]
        for col in date_cols:
            if col not in out.columns:
                out[col] = pd.NaT

        out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.normalize()

        num_cols = [
            "headline_sentiment", "summary_sentiment", "combined_sentiment",
            "sentiment_neg", "sentiment_neu", "sentiment_pos"
        ]
        for col in num_cols:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

        out = out.dropna(subset=["Date"]).reset_index(drop=True)

        keep_cols = [
            "published_at", "Date", "Ticker", "headline", "summary", "source", "url",
            "headline_sentiment", "summary_sentiment", "combined_sentiment",
            "sentiment_neg", "sentiment_neu", "sentiment_pos"
        ]
        out = out[keep_cols].copy()

        return out

    def fetch_finnhub_company_news(
        self,
        ticker: str,
        date_from: str,
        date_to: str
    ) -> pd.DataFrame:
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": date_from,
            "to": date_to,
            "token": self.api_key
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) == 0:
            return self._standardize_news_df(pd.DataFrame())

        rows = []
        for item in data:
            ts = item.get("datetime")
            published_at = pd.to_datetime(ts, unit="s", utc=True) if ts else pd.NaT

            rows.append({
                "published_at": published_at,
                "Date": published_at.normalize() if pd.notnull(published_at) else pd.NaT,
                "Ticker": str(ticker),
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", "finnhub"),
                "url": item.get("url", ""),
                "headline_sentiment": 0.0,
                "summary_sentiment": 0.0,
                "combined_sentiment": 0.0,
                "sentiment_neg": 0.0,
                "sentiment_neu": 0.0,
                "sentiment_pos": 0.0,
            })

        return self._standardize_news_df(pd.DataFrame(rows))

    def fetch_last_n_days(self, n_days: int = 7) -> pd.DataFrame:
        date_to = datetime.now(UTC).date()
        date_from = date_to - timedelta(days=n_days)

        all_news = []
        for ticker in self.train_universe:
            df = self.fetch_finnhub_company_news(
                ticker=ticker,
                date_from=str(date_from),
                date_to=str(date_to)
            )
            all_news.append(df)

        if len(all_news) == 0:
            return self._standardize_news_df(pd.DataFrame())

        out = pd.concat(all_news, axis=0, ignore_index=True)
        out = self._standardize_news_df(out)
        out = out.drop_duplicates(subset=["Ticker", "published_at", "headline"]).reset_index(drop=True)
        return out

    def append_to_existing_news(
        self,
        historical_news_df: pd.DataFrame,
        new_news_df: pd.DataFrame
    ) -> pd.DataFrame:
        hist = self._standardize_news_df(historical_news_df)
        new = self._standardize_news_df(new_news_df)

        out = pd.concat([hist, new], axis=0, ignore_index=True)
        out = out.drop_duplicates(subset=["Ticker", "published_at", "headline"])
        out = out.sort_values(["Ticker", "published_at", "headline"]).reset_index(drop=True)
        return out