# src/feature_engineer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class FeatureEngineer:
    universe: List[str]
    market_context: List[str]

    def build_stock_specific_feature_store(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build one wide dataframe per stock, matching the old AAPL-style architecture,
        then stack them into one long dataframe with stock-specific column names preserved.
        """
        all_frames = []

        for stock in self.universe:
            cols_to_keep = [
                "Date",
                f"Close_{stock}", f"Volume_{stock}",
                "Close_SPY", "Volume_SPY",
                "Close_QQQ", "Volume_QQQ",
                "Close_GLD", "Volume_GLD",
                "Close_TLT", "Volume_TLT",
                "Close_VIX",
                "Close_USO", "Volume_USO",
            ]

            missing_cols = [c for c in cols_to_keep if c not in raw_df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for {stock}: {missing_cols}")

            df = raw_df[cols_to_keep].copy()
            df = df.sort_values("Date").reset_index(drop=True)

            assets = [stock, "SPY", "QQQ", "GLD", "TLT", "USO"]

            for asset in assets:
                # returns
                df[f"ret_1d_{asset}"] = df[f"Close_{asset}"].pct_change(1, fill_method=None)
                df[f"ret_3d_{asset}"] = df[f"Close_{asset}"].pct_change(3, fill_method=None)
                df[f"ret_5d_{asset}"] = df[f"Close_{asset}"].pct_change(5, fill_method=None)
                df[f"ret_10d_{asset}"] = df[f"Close_{asset}"].pct_change(10, fill_method=None)
                df[f"ret_20d_{asset}"] = df[f"Close_{asset}"].pct_change(20, fill_method=None)
                df[f"ret_60d_{asset}"] = df[f"Close_{asset}"].pct_change(60, fill_method=None)

                # moving averages
                df[f"ma_5_{asset}"] = df[f"Close_{asset}"].rolling(5).mean()
                df[f"ma_10_{asset}"] = df[f"Close_{asset}"].rolling(10).mean()
                df[f"ma_20_{asset}"] = df[f"Close_{asset}"].rolling(20).mean()
                df[f"ma_50_{asset}"] = df[f"Close_{asset}"].rolling(50).mean()
                df[f"ma_100_{asset}"] = df[f"Close_{asset}"].rolling(100).mean()
                df[f"ma_200_{asset}"] = df[f"Close_{asset}"].rolling(200).mean()

                # price vs moving average
                df[f"price_vs_ma20_{asset}"] = df[f"Close_{asset}"] / df[f"ma_20_{asset}"]
                df[f"price_vs_ma50_{asset}"] = df[f"Close_{asset}"] / df[f"ma_50_{asset}"]
                df[f"price_vs_ma200_{asset}"] = df[f"Close_{asset}"] / df[f"ma_200_{asset}"]

                # rolling volatility
                df[f"vol_5_{asset}"] = df[f"ret_1d_{asset}"].rolling(5).std()
                df[f"vol_10_{asset}"] = df[f"ret_1d_{asset}"].rolling(10).std()
                df[f"vol_20_{asset}"] = df[f"ret_1d_{asset}"].rolling(20).std()
                df[f"vol_60_{asset}"] = df[f"ret_1d_{asset}"].rolling(60).std()

                # volume features if available
                if f"Volume_{asset}" in df.columns:
                    df[f"volchg_1d_{asset}"] = df[f"Volume_{asset}"].pct_change(1, fill_method=None)
                    df[f"volavg_5_{asset}"] = df[f"Volume_{asset}"].rolling(5).mean()
                    df[f"volavg_20_{asset}"] = df[f"Volume_{asset}"].rolling(20).mean()
                # momentum regime / trend strength
                df[f"momentum_regime_{asset}"] = (
                    (df[f"ret_20d_{asset}"] > 0) & (df[f"ret_60d_{asset}"] > 0)
                ).astype(float)

                # volatility regime
                df[f"vol_ratio_20_60_{asset}"] = df[f"vol_20_{asset}"] / df[f"vol_60_{asset}"]

                # z-score style distance from trend
                df[f"zscore_20_{asset}"] = (
                    (df[f"Close_{asset}"] - df[f"ma_20_{asset}"]) / df[f"vol_20_{asset}"]
                )
                df[f"zscore_60_{asset}"] = (
                    (df[f"Close_{asset}"] - df[f"ma_50_{asset}"]) / df[f"vol_60_{asset}"]
                )

                # distance from rolling extremes
                roll_max_20 = df[f"Close_{asset}"].rolling(20).max()
                roll_min_20 = df[f"Close_{asset}"].rolling(20).min()
                roll_max_60 = df[f"Close_{asset}"].rolling(60).max()
                roll_min_60 = df[f"Close_{asset}"].rolling(60).min()

                df[f"drawdown_20_{asset}"] = df[f"Close_{asset}"] / roll_max_20 - 1
                df[f"drawdown_60_{asset}"] = df[f"Close_{asset}"] / roll_max_60 - 1
                df[f"distance_from_low_20_{asset}"] = df[f"Close_{asset}"] / roll_min_20 - 1
                df[f"distance_from_low_60_{asset}"] = df[f"Close_{asset}"] / roll_min_60 - 1

                # breakout features
                df[f"breakout_20_{asset}"] = df[f"Close_{asset}"] / roll_max_20
                df[f"breakout_60_{asset}"] = df[f"Close_{asset}"] / roll_max_60
            # VIX-specific features
            df["ret_1d_VIX"] = df["Close_VIX"].pct_change(1, fill_method=None)
            df["ret_5d_VIX"] = df["Close_VIX"].pct_change(5, fill_method=None)
            df["ret_10d_VIX"] = df["Close_VIX"].pct_change(10, fill_method=None)
            df["ma_5_VIX"] = df["Close_VIX"].rolling(5).mean()
            df["ma_20_VIX"] = df["Close_VIX"].rolling(20).mean()
            df["ma_60_VIX"] = df["Close_VIX"].rolling(60).mean()
            df["price_vs_ma20_VIX"] = df["Close_VIX"] / df["ma_20_VIX"]
            df["price_vs_ma60_VIX"] = df["Close_VIX"] / df["ma_60_VIX"]
            df["vol_10_VIX"] = df["ret_1d_VIX"].rolling(10).std()
            df["vol_20_VIX"] = df["ret_1d_VIX"].rolling(20).std()

            stock_lower = stock.lower()

            # stock-relative features with stock-specific names
            df[f"{stock_lower}_vs_spy_5d"] = df[f"ret_5d_{stock}"] - df["ret_5d_SPY"]
            df[f"{stock_lower}_vs_qqq_5d"] = df[f"ret_5d_{stock}"] - df["ret_5d_QQQ"]
            df[f"{stock_lower}_vs_spy_20d"] = df[f"ret_20d_{stock}"] - df["ret_20d_SPY"]
            df[f"{stock_lower}_vs_qqq_20d"] = df[f"ret_20d_{stock}"] - df["ret_20d_QQQ"]
            df[f"{stock_lower}_vs_spy_60d"] = df[f"ret_60d_{stock}"] - df["ret_60d_SPY"]
            df[f"{stock_lower}_vs_qqq_60d"] = df[f"ret_60d_{stock}"] - df["ret_60d_QQQ"]

            df["risk_off_signal"] = df["ret_5d_GLD"] + df["ret_5d_TLT"] - df["ret_5d_SPY"]
            df["growth_risk_signal"] = df["ret_5d_QQQ"] - df["ret_5d_TLT"]
            df["oil_equity_signal"] = df["ret_5d_USO"] - df["ret_5d_SPY"]
            df["qqq_vs_spy_20d"] = df["ret_20d_QQQ"] - df["ret_20d_SPY"]
            df["qqq_vs_spy_60d"] = df["ret_60d_QQQ"] - df["ret_60d_SPY"]
            df["gld_vs_spy_20d"] = df["ret_20d_GLD"] - df["ret_20d_SPY"]
            df["tlt_vs_spy_20d"] = df["ret_20d_TLT"] - df["ret_20d_SPY"]
            df["uso_vs_spy_20d"] = df["ret_20d_USO"] - df["ret_20d_SPY"]
            df["vol_regime_high"] = (df["Close_VIX"] > df["ma_20_VIX"]).astype(float)
            df[f"trend_regime_{stock_lower}"] = (df[f"Close_{stock}"] > df[f"ma_50_{stock}"]).astype(float)
            df["trend_regime_spy"] = (df["Close_SPY"] > df["ma_50_SPY"]).astype(float)

            # generic helpers for downstream code
            df["Ticker"] = stock
            df["Close"] = df[f"Close_{stock}"]
            df["Volume"] = df[f"Volume_{stock}"]

            all_frames.append(df)

        features_df = pd.concat(all_frames, axis=0, ignore_index=True)
        features_df = features_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

        return features_df

    def validate_feature_store(self, df: pd.DataFrame) -> None:
        required_cols = ["Date", "Ticker", "Close", "Volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required feature-store columns: {missing_cols}")

        if len(df) == 0:
            raise ValueError("Feature store is empty.")