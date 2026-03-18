# src/evaluator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class ModelEvaluator:
    confidence_thresholds: List[float]

    def evaluate_basic(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "n_rows": len(df),
            "accuracy": accuracy_score(df["target_class"], df["predicted_class"]),
            "macro_precision": precision_score(df["target_class"], df["predicted_class"], average="macro", zero_division=0),
            "macro_recall": recall_score(df["target_class"], df["predicted_class"], average="macro", zero_division=0),
            "macro_f1": f1_score(df["target_class"], df["predicted_class"], average="macro", zero_division=0),
        }

    def add_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["confidence"] = out["meta_probability"]
        return out

    def evaluate_by_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for ticker, sub in df.groupby("Ticker"):
            metrics = self.evaluate_basic(sub)
            naive_acc = (sub["naive_prediction_class"] == sub["target_class"]).mean()

            rows.append({
                "Ticker": ticker,
                "n_rows": metrics["n_rows"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "naive_accuracy": naive_acc,
                "accuracy_lift": metrics["accuracy"] - naive_acc,
            })

        return pd.DataFrame(rows).sort_values("accuracy_lift", ascending=False).reset_index(drop=True)

    def evaluate_confidence_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_confidence(df)
        rows = []

        for threshold in self.confidence_thresholds:
            sub = df[(df["confidence"] >= threshold) & (df["selected_trade"] == 1)].copy()
            coverage = len(sub) / len(df) if len(df) > 0 else np.nan

            if len(sub) == 0:
                rows.append({
                    "threshold": threshold,
                    "n_rows": 0,
                    "coverage": 0.0,
                    "accuracy": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall": np.nan,
                    "macro_f1": np.nan,
                    "naive_accuracy": np.nan,
                    "accuracy_lift": np.nan,
                })
                continue

            metrics = self.evaluate_basic(sub)
            naive_acc = (sub["naive_prediction_class"] == sub["target_class"]).mean()

            rows.append({
                "threshold": threshold,
                "n_rows": len(sub),
                "coverage": coverage,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "naive_accuracy": naive_acc,
                "accuracy_lift": metrics["accuracy"] - naive_acc,
            })

        return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    def evaluate_confidence_by_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_confidence(df)
        rows = []

        for ticker, stock_df in df.groupby("Ticker"):
            for threshold in self.confidence_thresholds:
                sub = stock_df[(stock_df["confidence"] >= threshold) & (stock_df["selected_trade"] == 1)].copy()
                coverage = len(sub) / len(stock_df) if len(stock_df) > 0 else np.nan

                if len(sub) == 0:
                    rows.append({
                        "Ticker": ticker,
                        "threshold": threshold,
                        "n_rows": 0,
                        "coverage": 0.0,
                        "accuracy": np.nan,
                        "macro_precision": np.nan,
                        "macro_recall": np.nan,
                        "macro_f1": np.nan,
                        "naive_accuracy": np.nan,
                        "accuracy_lift": np.nan,
                    })
                    continue

                metrics = self.evaluate_basic(sub)
                naive_acc = (sub["naive_prediction_class"] == sub["target_class"]).mean()

                rows.append({
                    "Ticker": ticker,
                    "threshold": threshold,
                    "n_rows": len(sub),
                    "coverage": coverage,
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "naive_accuracy": naive_acc,
                    "accuracy_lift": metrics["accuracy"] - naive_acc,
                })

        return pd.DataFrame(rows).sort_values(
            ["threshold", "accuracy_lift"],
            ascending=[True, False]
        ).reset_index(drop=True)

    def evaluate_time_segments(self, df: pd.DataFrame, n_segments: int = 10) -> pd.DataFrame:
        df = df[df["selected_trade"] == 1].sort_values("Date").reset_index(drop=True)

        if len(df) == 0:
            return pd.DataFrame()

        segment_size = int(len(df) / n_segments)
        rows = []

        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(df)

            segment_df = df.iloc[start_idx:end_idx].copy()
            if len(segment_df) == 0:
                continue

            metrics = self.evaluate_basic(segment_df)
            naive_acc = (segment_df["naive_prediction_class"] == segment_df["target_class"]).mean()

            rows.append({
                "segment": i + 1,
                "start_date": segment_df["Date"].min(),
                "end_date": segment_df["Date"].max(),
                "n_rows": len(segment_df),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "naive_accuracy": naive_acc,
                "accuracy_lift": metrics["accuracy"] - naive_acc,
            })

        return pd.DataFrame(rows)

    def backtest_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        traded = df[df["selected_trade"] == 1].copy()

        if len(traded) == 0:
            return pd.DataFrame([{
                "n_trades": 0,
                "coverage": 0.0,
                "hit_rate": np.nan,
                "mean_return": np.nan,
                "volatility": np.nan,
                "sharpe_like": np.nan,
                "cumulative_return": np.nan
            }])

        traded = traded.sort_values("Date").copy()

        mean_ret = traded["strategy_return"].mean()
        vol = traded["strategy_return"].std()
        sharpe_like = mean_ret / vol if pd.notnull(vol) and vol > 0 else np.nan
        cum_ret = traded["strategy_return"].cumsum().iloc[-1]
        return pd.DataFrame([{
            "n_trades": len(traded),
            "coverage": len(traded) / len(df),
            "hit_rate": (traded["strategy_return"] > 0).mean(),
            "mean_return": mean_ret,
            "volatility": vol,
            "sharpe_like": sharpe_like,
            "cumulative_return": cum_ret
        }])

    def backtest_by_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for ticker, sub in df.groupby("Ticker"):
            traded = sub[sub["selected_trade"] == 1].copy()

            if len(traded) == 0:
                rows.append({
                    "Ticker": ticker,
                    "n_trades": 0,
                    "coverage": 0.0,
                    "hit_rate": np.nan,
                    "mean_return": np.nan,
                    "volatility": np.nan,
                    "sharpe_like": np.nan,
                    "cumulative_return": np.nan
                })
                continue

            mean_ret = traded["strategy_return"].mean()
            vol = traded["strategy_return"].std()
            sharpe_like = mean_ret / vol if pd.notnull(vol) and vol > 0 else np.nan
            cum_ret = (1 + traded["strategy_return"]).prod() - 1

            rows.append({
                "Ticker": ticker,
                "n_trades": len(traded),
                "coverage": len(traded) / len(sub),
                "hit_rate": (traded["strategy_return"] > 0).mean(),
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe_like": sharpe_like,
                "cumulative_return": cum_ret
            })

        return pd.DataFrame(rows).sort_values("sharpe_like", ascending=False).reset_index(drop=True)