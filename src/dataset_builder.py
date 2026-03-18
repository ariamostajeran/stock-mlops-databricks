# src/dataset_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


@dataclass
class DatasetBuilder:
    target_horizon_days: int
    target_return_threshold: float

    def add_targets(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        out_frames = []

        for ticker, sub in df.groupby("Ticker", sort=False):
            sub = sub.sort_values("Date").reset_index(drop=True)

            sub["future_close"] = sub["Close"].shift(-self.target_horizon_days)
            sub["future_return"] = sub["future_close"] / sub["Close"] - 1

            # keep binary target too
            sub["target_up_down"] = (sub["future_close"] > sub["Close"]).astype(int)

            # new 3-class target
            sub["target_class"] = np.select(
                [
                    sub["future_return"] > self.target_return_threshold,
                    sub["future_return"] < -self.target_return_threshold
                ],
                [2, 0],   # 2=UP, 0=DOWN
                default=1 # 1=FLAT
            ).astype(int)

            out_frames.append(sub)

        out = pd.concat(out_frames, axis=0, ignore_index=True)
        out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        return out

    def add_naive_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        naive_bin = []
        naive_cls = []

        for ticker, sub in out.groupby("Ticker", sort=False):
            sub = sub.copy()

            ret5 = sub["Close"].pct_change(5, fill_method=None)

            naive_bin.append((ret5 > 0).astype(int))

            naive_cls.append(
                np.select(
                    [
                        ret5 > self.target_return_threshold,
                        ret5 < -self.target_return_threshold
                    ],
                    [2, 0],
                    default=1
                )
            )

        out["naive_prediction"] = pd.concat(naive_bin, axis=0).sort_index().values
        out["naive_prediction_class"] = np.concatenate(naive_cls)

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

        # For stock-specific wide architecture, do NOT drop using all global feature columns,
        # because each stock row legitimately has NaNs in other stocks' columns.
        # Only require the core target columns here.
        ml_df = ml_df.dropna(
            subset=["future_close", "future_return", "target_class", "naive_prediction_class"]
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
            "target_class",
            "naive_prediction",
            "naive_prediction_class",
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
            elif any(
                col.startswith(prefix) for prefix in [
                    "Close_SPY", "Volume_SPY", "Close_QQQ", "Volume_QQQ", "Close_GLD", "Volume_GLD",
                    "Close_TLT", "Volume_TLT", "Close_VIX", "Close_USO", "Volume_USO",
                    "ret_", "ma_", "price_vs_ma", "vol_", "volchg_", "volavg_",
                    "risk_off_signal", "growth_risk_signal", "oil_equity_signal",
                    "qqq_vs_spy_20d", "qqq_vs_spy_60d", "gld_vs_spy_20d", "tlt_vs_spy_20d", "uso_vs_spy_20d",
                    "vol_regime_high", "trend_regime_spy"
                ]
            ):
                # shared features are okay; stock-specific junk will be filtered by absence below
                if (
                    f"_{ticker_upper}" not in col and
                    not any(f"_{other}" in col for other in ["AAPL","AMD","AMZN","GOOGL","INTC","META","MSFT","NFLX","NVDA","TSLA"])
                ) or col in ["Close", "Volume"]:
                    keep_cols.append(col)

        keep_cols += [c for c in feature_cols if c in ["Close", "Volume"]]
        return sorted(list(set(keep_cols)))

    def validate_ml_dataset(self, df: pd.DataFrame) -> None:
        required_cols = ["Date", "Ticker", "Close", "target_class", "naive_prediction_class"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required ML dataset columns: {missing_cols}")

        if len(df) == 0:
            raise ValueError("ML dataset is empty.")