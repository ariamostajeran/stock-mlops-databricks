# src/inference_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd


@dataclass
class InferencePipeline:
    def score_latest(
        self,
        ml_df: pd.DataFrame,
        tickers: List[str],
        final_models: Dict[str, Any],
        final_feature_sets: Dict[str, List[str]]
    ) -> pd.DataFrame:
        rows = []

        for ticker in tickers:
            if ticker not in final_models or ticker not in final_feature_sets:
                continue

            sub = ml_df[ml_df["Ticker"] == ticker].copy().sort_values("Date")
            if len(sub) == 0:
                continue

            latest_row = sub.iloc[[-1]].copy()
            feature_cols = final_feature_sets[ticker]
            X_latest = latest_row[feature_cols].copy()

            model = final_models[ticker]
            prob_up = model.predict_proba(X_latest)[:, 1][0]
            pred = int(prob_up >= 0.5)

            rows.append({
                "Date": latest_row["Date"].iloc[0],
                "Ticker": ticker,
                "predicted_probability": prob_up,
                "predicted_direction": pred
            })

        return pd.DataFrame(rows)