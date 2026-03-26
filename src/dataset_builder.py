# src/dataset_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


@dataclass
class DatasetBuilder:
    target_horizon_days: int

    def add_targets(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        out_frames = []

        for ticker, sub in df.groupby("Ticker", sort=False):
            sub = sub.sort_values("Date").reset_index(drop=True)

            sub["future_close"] = sub["Close"].shift(-self.target_horizon_days)
            sub["future_return"] = sub["future_close"] / sub["Close"] - 1
            sub["target_up_down"] = (sub["future_close"] > sub["Close"]).astype(int)

            out_frames.append(sub)

        out = pd.concat(out_frames, axis=0, ignore_index=True)
        out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        return out

    def add_naive_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        naive_preds = []
        for _, sub in out.groupby("Ticker", sort=False):
            ret5 = sub["Close"].pct_change(5, fill_method=None)
            naive_preds.append((ret5 > 0).astype(int))

        out["naive_prediction"] = pd.concat(naive_preds, axis=0).sort_index().values
        return out

    def sanitize_numeric_values(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return out

    def build_ml_dataset(
        self,
        df: pd.DataFrame,
        feature_exclude_cols: List[str] | None = None
    ) -> pd.DataFrame:
        if feature_exclude_cols is None:
            feature_exclude_cols = []

        ml_df = self.sanitize_numeric_values(df.copy())
        ml_df = ml_df.dropna(
            subset=["future_close", "future_return", "target_up_down", "naive_prediction"]
        ).reset_index(drop=True)

        return ml_df

    def get_feature_columns(
        self,
        df: pd.DataFrame,
        ticker: str | None = None,
        feature_exclude_cols: List[str] | None = None
    ) -> List[str]:
        if feature_exclude_cols is None:
            feature_exclude_cols = []

        exclude_cols = set([
            "Date",
            "Ticker",
            "future_close",
            "future_return",
            "target_up_down",
            "naive_prediction",
        ] + feature_exclude_cols)

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        if ticker is None:
            return feature_cols

        ticker_upper = ticker.upper()
        ticker_lower = ticker.lower()

        keep_cols = []
        for col in feature_cols:
            if f"_{ticker_upper}" in col or col.startswith(f"{ticker_lower}_vs_") or col == f"trend_regime_{ticker_lower}":
                keep_cols.append(col)
            elif (
                ("SPY" in col) or ("QQQ" in col) or ("GLD" in col) or ("TLT" in col) or
                ("USO" in col) or ("VIX" in col) or
                ("risk_off_signal" in col) or ("growth_risk_signal" in col) or
                ("oil_equity_signal" in col) or ("news" in col.lower()) or
                ("sentiment" in col.lower()) or ("source_count" in col.lower())
            ):
                keep_cols.append(col)
            elif col in ["Close", "Volume"]:
                keep_cols.append(col)

        return sorted(list(set(keep_cols)))

    def validate_ml_dataset(self, df: pd.DataFrame) -> None:
        required_cols = ["Date", "Ticker", "Close", "target_up_down", "naive_prediction"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required ML dataset columns: {missing_cols}")

        if len(df) == 0:
            raise ValueError("ML dataset is empty.")