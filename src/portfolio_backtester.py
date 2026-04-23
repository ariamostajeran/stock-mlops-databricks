# src/portfolio_backtester.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class PortfolioBacktester:
    transaction_cost_per_trade: float = 0.0005
    max_positions_per_day: Optional[int] = None
    weighting_method: str = "equal"   # equal | confidence

    def _apply_position_sizing(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        traded = out[out["selected_trade"] == 1].copy()
        if len(traded) == 0:
            out["position_weight"] = 0.0
            return out

        weights = []

        for date, sub in out.groupby("Date", sort=True):
            sub = sub.copy()
            active = sub[sub["selected_trade"] == 1].copy()

            if len(active) == 0:
                sub["position_weight"] = 0.0
                weights.append(sub)
                continue

            if self.max_positions_per_day is not None and len(active) > self.max_positions_per_day:
                active = active.sort_values("signal_confidence", ascending=False).head(self.max_positions_per_day)

            selected_idx = active.index.tolist()

            sub["selected_trade"] = 0
            sub.loc[selected_idx, "selected_trade"] = 1

            active = sub[sub["selected_trade"] == 1].copy()

            if self.weighting_method == "confidence":
                denom = active["signal_confidence"].sum()
                if denom > 0:
                    active["position_weight"] = active["signal_confidence"] / denom
                else:
                    active["position_weight"] = 1.0 / len(active)
            else:
                active["position_weight"] = 1.0 / len(active)

            sub["position_weight"] = 0.0
            sub.loc[active.index, "position_weight"] = active["position_weight"]

            weights.append(sub)

        out = pd.concat(weights, axis=0).sort_values(["Date", "Ticker"]).reset_index(drop=True)
        return out

    def backtest(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        df = signals_df.copy()
        df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

        df = self._apply_position_sizing(df)

        df["gross_return"] = df["signal"] * df["future_return"] * df["position_weight"]
        df["transaction_cost"] = np.where(df["selected_trade"] == 1, self.transaction_cost_per_trade * df["position_weight"], 0.0)
        df["net_return"] = df["gross_return"] - df["transaction_cost"]

        daily = (
            df.groupby("Date", as_index=False)
            .agg(
                n_positions=("selected_trade", "sum"),
                gross_return=("gross_return", "sum"),
                transaction_cost=("transaction_cost", "sum"),
                portfolio_return=("net_return", "sum"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        capital = 1.0
        capitals = []
        for r in daily["portfolio_return"]:
            capital *= (1 + r)
            capitals.append(capital)

        daily["equity_curve"] = capitals
        daily["cumulative_return"] = daily["equity_curve"] - 1.0
        daily["rolling_peak"] = daily["equity_curve"].cummax()
        daily["drawdown"] = daily["equity_curve"] / daily["rolling_peak"] - 1.0

        return daily

    def summary(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        df = portfolio_df.copy()

        if len(df) == 0:
            return pd.DataFrame([{
                "n_days": 0,
                "mean_daily_return": np.nan,
                "volatility": np.nan,
                "sharpe_like": np.nan,
                "hit_rate": np.nan,
                "max_drawdown": np.nan,
                "final_cumulative_return": np.nan,
                "avg_positions_per_day": np.nan,
            }])

        mean_ret = df["portfolio_return"].mean()
        vol = df["portfolio_return"].std()
        sharpe_like = mean_ret / vol if pd.notnull(vol) and vol > 0 else np.nan

        return pd.DataFrame([{
            "n_days": len(df),
            "mean_daily_return": mean_ret,
            "volatility": vol,
            "sharpe_like": sharpe_like,
            "hit_rate": (df["portfolio_return"] > 0).mean(),
            "max_drawdown": df["drawdown"].min(),
            "final_cumulative_return": df["cumulative_return"].iloc[-1],
            "avg_positions_per_day": df["n_positions"].mean(),
        }])

    def by_stock(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        df = signals_df.copy()
        traded = df[df["selected_trade"] == 1].copy()

        rows = []
        for ticker, sub in traded.groupby("Ticker"):
            mean_ret = sub["future_return"].mul(sub["signal"]).mean()
            vol = sub["future_return"].mul(sub["signal"]).std()
            sharpe_like = mean_ret / vol if pd.notnull(vol) and vol > 0 else np.nan

            rows.append({
                "Ticker": ticker,
                "n_trades": len(sub),
                "hit_rate": ((sub["signal"] * sub["future_return"]) > 0).mean(),
                "mean_trade_return": mean_ret,
                "trade_volatility": vol,
                "sharpe_like": sharpe_like,
                "avg_confidence": sub["signal_confidence"].mean(),
            })

        return pd.DataFrame(rows).sort_values("sharpe_like", ascending=False).reset_index(drop=True)