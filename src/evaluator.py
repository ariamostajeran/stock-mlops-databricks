# src/evaluator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


@dataclass
class ModelEvaluator:
    confidence_thresholds: List[float]

    def evaluate_basic(self, df: pd.DataFrame) -> Dict[str, Any]:
        auc = roc_auc_score(df["target_up_down"], df["predicted_probability"]) if df["target_up_down"].nunique() > 1 else np.nan

        return {
            "n_rows": len(df),
            "accuracy": accuracy_score(df["target_up_down"], df["predicted_direction"]),
            "precision": precision_score(df["target_up_down"], df["predicted_direction"], zero_division=0),
            "recall": recall_score(df["target_up_down"], df["predicted_direction"], zero_division=0),
            "auc": auc,
        }

    def evaluate_by_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for ticker, sub in df.groupby("Ticker"):
            metrics = self.evaluate_basic(sub)
            naive_acc = (sub["naive_prediction"] == sub["target_up_down"]).mean()

            rows.append({
                "Ticker": ticker,
                "n_rows": metrics["n_rows"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auc": metrics["auc"],
                "naive_accuracy": naive_acc,
                "accuracy_lift": metrics["accuracy"] - naive_acc,
            })

        return pd.DataFrame(rows).sort_values("accuracy_lift", ascending=False).reset_index(drop=True)

    def evaluate_confidence_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for threshold in self.confidence_thresholds:
            sub = df[df["predicted_probability"] >= threshold].copy()
            coverage = len(sub) / len(df) if len(df) > 0 else np.nan

            if len(sub) == 0:
                rows.append({
                    "threshold": threshold,
                    "n_rows": 0,
                    "coverage": 0.0,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "auc": np.nan,
                    "naive_accuracy": np.nan,
                    "accuracy_lift": np.nan,
                })
                continue

            metrics = self.evaluate_basic(sub)
            naive_acc = (sub["naive_prediction"] == sub["target_up_down"]).mean()

            rows.append({
                "threshold": threshold,
                "n_rows": len(sub),
                "coverage": coverage,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auc": metrics["auc"],
                "naive_accuracy": naive_acc,
                "accuracy_lift": metrics["accuracy"] - naive_acc,
            })

        return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    def evaluate_confidence_by_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for ticker, stock_df in df.groupby("Ticker"):

            for threshold in self.confidence_thresholds:

                # ===== UP SIDE =====
                up_df = stock_df[stock_df["predicted_probability"] >= threshold].copy()
                up_coverage = len(up_df) / len(stock_df) if len(stock_df) > 0 else np.nan

                if len(up_df) > 0:
                    up_metrics = self.evaluate_basic(up_df)
                    up_naive = (up_df["naive_prediction"] == up_df["target_up_down"]).mean()
                    up_lift = up_metrics["accuracy"] - up_naive
                else:
                    up_metrics = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "auc": np.nan}
                    up_naive = np.nan
                    up_lift = np.nan

                # ===== DOWN SIDE =====
                down_df = stock_df[stock_df["predicted_probability"] <= (1 - threshold)].copy()
                down_coverage = len(down_df) / len(stock_df) if len(stock_df) > 0 else np.nan

                if len(down_df) > 0:
                    down_metrics = self.evaluate_basic(down_df)
                    down_naive = (down_df["naive_prediction"] == down_df["target_up_down"]).mean()
                    down_lift = down_metrics["accuracy"] - down_naive
                else:
                    down_metrics = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "auc": np.nan}
                    down_naive = np.nan
                    down_lift = np.nan

                rows.append({
                    "Ticker": ticker,
                    "threshold": threshold,

                    # ----- UP -----
                    "up_n_rows": len(up_df),
                    "up_coverage": up_coverage,
                    "accuracy_up": up_metrics["accuracy"],
                    "precision_up": up_metrics["precision"],
                    "recall_up": up_metrics["recall"],
                    "auc_up": up_metrics["auc"],
                    "naive_accuracy_up": up_naive,
                    "accuracy_lift_up": up_lift,

                    # ----- DOWN -----
                    "down_n_rows": len(down_df),
                    "down_coverage": down_coverage,
                    "accuracy_down": down_metrics["accuracy"],
                    "precision_down": down_metrics["precision"],
                    "recall_down": down_metrics["recall"],
                    "auc_down": down_metrics["auc"],
                    "naive_accuracy_down": down_naive,
                    "accuracy_lift_down": down_lift,
                })

        return pd.DataFrame(rows).sort_values(
            ["threshold", "accuracy_lift_up"],
            ascending=[True, False]
        ).reset_index(drop=True)

    def evaluate_time_segments(self, df: pd.DataFrame, n_segments: int = 10) -> pd.DataFrame:
        df = df.sort_values("Date").reset_index(drop=True)

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
            naive_acc = (segment_df["naive_prediction"] == segment_df["target_up_down"]).mean()

            rows.append({
                "segment": i + 1,
                "start_date": segment_df["Date"].min(),
                "end_date": segment_df["Date"].max(),
                "n_rows": len(segment_df),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auc": metrics["auc"],
                "naive_accuracy": naive_acc,
                "accuracy_lift": metrics["accuracy"] - naive_acc,
            })

        return pd.DataFrame(rows)

    def backtest_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        traded = df.copy()

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

        capital = 1.0
        for r in traded["strategy_return"]:
            capital *= (1 + r)
        cum_ret = capital - 1

        return pd.DataFrame([{
            "n_trades": len(traded),
            "coverage": 1.0,
            "hit_rate": (traded["strategy_return"] > 0).mean(),
            "mean_return": mean_ret,
            "volatility": vol,
            "sharpe_like": sharpe_like,
            "cumulative_return": cum_ret
        }])

    def backtest_by_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for ticker, sub in df.groupby("Ticker"):
            traded = sub.copy()

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

            capital = 1.0
            for r in traded["strategy_return"]:
                capital *= (1 + r)
            cum_ret = capital - 1

            rows.append({
                "Ticker": ticker,
                "n_trades": len(traded),
                "coverage": 1.0,
                "hit_rate": (traded["strategy_return"] > 0).mean(),
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe_like": sharpe_like,
                "cumulative_return": cum_ret
            })

        return pd.DataFrame(rows).sort_values("sharpe_like", ascending=False).reset_index(drop=True)