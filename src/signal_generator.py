# src/signal_generator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class SignalGenerator:
    up_thresholds: Optional[Dict[str, float]] = None
    down_thresholds: Optional[Dict[str, float]] = None
    default_up_threshold: float = 0.60
    default_down_threshold: float = 0.40

    def _get_up_threshold(self, ticker: str) -> float:
        if self.up_thresholds is None:
            return self.default_up_threshold
        return self.up_thresholds.get(ticker, self.default_up_threshold)

    def _get_down_threshold(self, ticker: str) -> float:
        if self.down_thresholds is None:
            return self.default_down_threshold
        return self.down_thresholds.get(ticker, self.default_down_threshold)

    def generate_signals(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        df = predictions_df.copy()
        df["Ticker"] = df["Ticker"].astype(str)

        signal_rows = []

        for ticker, sub in df.groupby("Ticker", sort=False):
            sub = sub.sort_values("Date").copy()

            up_thr = self._get_up_threshold(ticker)
            down_thr = self._get_down_threshold(ticker)

            sub["up_threshold"] = up_thr
            sub["down_threshold"] = down_thr

            sub["signal"] = 0
            sub.loc[sub["predicted_probability"] >= up_thr, "signal"] = 1
            sub.loc[sub["predicted_probability"] <= down_thr, "signal"] = -1

            sub["signal_side"] = np.where(
                sub["signal"] == 1, "LONG",
                np.where(sub["signal"] == -1, "SHORT", "FLAT")
            )

            sub["signal_confidence"] = np.where(
                sub["signal"] == 1,
                sub["predicted_probability"],
                np.where(sub["signal"] == -1, 1 - sub["predicted_probability"], 0.0)
            )

            sub["selected_trade"] = (sub["signal"] != 0).astype(int)

            signal_rows.append(sub)

        out = pd.concat(signal_rows, axis=0, ignore_index=True)
        return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    def summarize_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        df = signals_df.copy()

        rows = []
        for ticker, sub in df.groupby("Ticker"):
            traded = sub[sub["selected_trade"] == 1].copy()

            rows.append({
                "Ticker": ticker,
                "n_rows": len(sub),
                "n_signals": len(traded),
                "signal_rate": len(traded) / len(sub) if len(sub) > 0 else np.nan,
                "n_long": (sub["signal"] == 1).sum(),
                "n_short": (sub["signal"] == -1).sum(),
                "avg_confidence": traded["signal_confidence"].mean() if len(traded) > 0 else np.nan,
            })

        return pd.DataFrame(rows).sort_values("signal_rate", ascending=False).reset_index(drop=True)

    def latest_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        df = signals_df.copy()
        latest_date = df["Date"].max()
        latest_df = df[df["Date"] == latest_date].copy()

        cols = [
            "Date",
            "Ticker",
            "predicted_probability",
            "signal",
            "signal_side",
            "signal_confidence",
            "up_threshold",
            "down_threshold",
        ]
        return latest_df[cols].sort_values(["signal_confidence", "Ticker"], ascending=[False, True]).reset_index(drop=True)