# src/reporting_monitoring.py

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ReportingMonitoring:
    min_accuracy_threshold: float = 0.52
    min_lift_threshold: float = 0.00
    min_signal_count: int = 20

    def model_health_report(
        self,
        eval_by_stock_df: pd.DataFrame,
        eval_confidence_by_stock_df: pd.DataFrame
    ) -> pd.DataFrame:
        stock_df = eval_by_stock_df.copy()

        if "accuracy_lift_up" in eval_confidence_by_stock_df.columns:
            best_conf = (
                eval_confidence_by_stock_df
                .groupby("Ticker", as_index=False)
                .agg(
                    best_up_lift=("accuracy_lift_up", "max"),
                    best_down_lift=("accuracy_lift_down", "max"),
                    best_up_accuracy=("accuracy_up", "max"),
                    best_down_accuracy=("accuracy_down", "max"),
                    best_up_n_rows=("up_n_rows", "max"),
                    best_down_n_rows=("down_n_rows", "max"),
                )
            )
            stock_df = stock_df.merge(best_conf, on="Ticker", how="left")

        stock_df["health_status"] = np.where(
            (stock_df["accuracy"] >= self.min_accuracy_threshold) &
            (stock_df["accuracy_lift"] >= self.min_lift_threshold),
            "OK",
            "CHECK"
        )

        return stock_df.sort_values(["health_status", "accuracy_lift"], ascending=[True, False]).reset_index(drop=True)

    def latest_signal_report(self, latest_signals_df: pd.DataFrame) -> pd.DataFrame:
        df = latest_signals_df.copy()

        df["action"] = np.where(
            df["signal"] == 1, "LONG",
            np.where(df["signal"] == -1, "SHORT", "NO_TRADE")
        )

        return df.sort_values(["signal_confidence", "Ticker"], ascending=[False, True]).reset_index(drop=True)

    def alert_flags(
        self,
        eval_by_stock_df: pd.DataFrame,
        signal_summary_df: pd.DataFrame,
        portfolio_summary_df: pd.DataFrame
    ) -> pd.DataFrame:
        alerts = []

        for _, row in eval_by_stock_df.iterrows():
            if row["accuracy"] < self.min_accuracy_threshold:
                alerts.append({
                    "alert_type": "low_accuracy",
                    "scope": row["Ticker"],
                    "alert_message": f'{row["Ticker"]} accuracy below threshold: {row["accuracy"]:.3f}'
                })

            if row["accuracy_lift"] < self.min_lift_threshold:
                alerts.append({
                    "alert_type": "negative_lift",
                    "scope": row["Ticker"],
                    "alert_message": f'{row["Ticker"]} accuracy lift below threshold: {row["accuracy_lift"]:.3f}'
                })

        for _, row in signal_summary_df.iterrows():
            if row["n_signals"] < self.min_signal_count:
                alerts.append({
                    "alert_type": "low_signal_count",
                    "scope": row["Ticker"],
                    "alert_message": f'{row["Ticker"]} has low signal count: {row["n_signals"]}'
                })

        if len(portfolio_summary_df) > 0:
            ps = portfolio_summary_df.iloc[0]
            if pd.notnull(ps["max_drawdown"]) and ps["max_drawdown"] < -0.20:
                alerts.append({
                    "alert_type": "high_drawdown",
                    "scope": "portfolio",
                    "alert_message": f'Portfolio max drawdown is high: {ps["max_drawdown"]:.3f}'
                })

        if len(alerts) == 0:
            alerts.append({
                "alert_type": "ok",
                "scope": "system",
                "alert_message": "No alert triggered."
            })

        return pd.DataFrame(alerts)

    def dashboard_summary(
        self,
        eval_by_stock_df: pd.DataFrame,
        portfolio_summary_df: pd.DataFrame,
        signal_summary_df: pd.DataFrame
    ) -> pd.DataFrame:
        portfolio = portfolio_summary_df.iloc[0] if len(portfolio_summary_df) > 0 else None

        return pd.DataFrame([{
            "n_stocks": len(eval_by_stock_df),
            "avg_stock_accuracy": eval_by_stock_df["accuracy"].mean() if len(eval_by_stock_df) > 0 else np.nan,
            "avg_stock_lift": eval_by_stock_df["accuracy_lift"].mean() if len(eval_by_stock_df) > 0 else np.nan,
            "total_signals": signal_summary_df["n_signals"].sum() if len(signal_summary_df) > 0 else 0,
            "avg_signal_rate": signal_summary_df["signal_rate"].mean() if len(signal_summary_df) > 0 else np.nan,
            "portfolio_final_return": portfolio["final_cumulative_return"] if portfolio is not None else np.nan,
            "portfolio_sharpe_like": portfolio["sharpe_like"] if portfolio is not None else np.nan,
            "portfolio_max_drawdown": portfolio["max_drawdown"] if portfolio is not None else np.nan,
        }])