# src/model_trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelTrainer:
    random_state: int = 42

    def prepare_single_ticker_dataset(
        self,
        ml_df: pd.DataFrame,
        feature_cols: List[str],
        ticker: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        df = ml_df[ml_df["Ticker"] == ticker].copy()
        df = df.sort_values("Date").reset_index(drop=True)
        model_feature_cols = [c for c in feature_cols if c != "Ticker"]
        return df, model_feature_cols

    def get_walk_forward_splits(
        self,
        df: pd.DataFrame,
        train_years: int,
        test_months: int
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        df = df.sort_values("Date").reset_index(drop=True).copy()
        min_date = pd.to_datetime(df["Date"].min())
        max_date = pd.to_datetime(df["Date"].max())

        splits = []

        train_start = min_date
        train_end = train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)

        while test_start < max_date:
            test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

            train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
            test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

            if len(train_df) > 200 and len(test_df) > 20:
                splits.append((train_df, test_df))

            train_end = train_end + pd.DateOffset(months=test_months)
            test_start = test_end + pd.DateOffset(days=1)

        return splits

    def fit_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> XGBClassifier:
        model = XGBClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=8,
            reg_alpha=3,
            min_child_weight=8,
            random_state=self.random_state,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss"
        )
        model.fit(X_train, y_train)
        return model

    def select_top_features(
        self,
        model: XGBClassifier,
        feature_cols: List[str],
        top_k: int
    ) -> List[str]:
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance.head(top_k)["feature"].tolist()

    def evaluate_predictions(
        self,
        y_true: pd.Series,
        y_pred
    ) -> Dict[str, Any]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def build_meta_dataset(
        self,
        df: pd.DataFrame,
        base_pred,
        base_prob,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        out = df.copy().reset_index(drop=True)

        out["predicted_class"] = base_pred
        out["pred_prob_down"] = base_prob[:, 0]
        out["pred_prob_flat"] = base_prob[:, 1]
        out["pred_prob_up"] = base_prob[:, 2]
        out["predicted_probability"] = np.max(base_prob, axis=1)

        trade_direction_map = {0: -1, 1: 0, 2: 1}
        out["trade_direction_base"] = out["predicted_class"].map(trade_direction_map)

        # Meta target: among non-flat base predictions, was the direction correct?
        out["meta_target"] = (
            (out["predicted_class"] != 1) &
            (out["predicted_class"] == out["target_class"])
        ).astype(int)

        meta_feature_cols = feature_cols + [
            "predicted_class",
            "pred_prob_down",
            "pred_prob_flat",
            "pred_prob_up",
            "predicted_probability",
        ]

        return out, meta_feature_cols

    def fit_meta_model(
        self,
        meta_df: pd.DataFrame,
        meta_feature_cols: List[str]
    ):
        eligible = meta_df[meta_df["predicted_class"] != 1].copy()

        if len(eligible) < 50:
            return None

        if eligible["meta_target"].nunique() < 2:
            return None

        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=self.random_state
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(eligible[meta_feature_cols])

        model.fit(X_scaled, eligible["meta_target"])

        model.scaler = scaler  # store it 
        return model

    def optimize_threshold(
        self,
        meta_df: pd.DataFrame,
        meta_prob,
        threshold_grid: List[float],
        min_trades: int
    ) -> Tuple[float, pd.DataFrame]:

        mask = (meta_df["predicted_class"] != 1)
        eligible = meta_df[mask].copy()
        eligible["meta_probability"] = meta_prob[mask]

        rows = []

        for thr in threshold_grid:
            sub = eligible[eligible["meta_probability"] >= thr].copy()

            if len(sub) < min_trades:
                rows.append({
                    "threshold": thr,
                    "n_trades": len(sub),
                    "hit_rate": np.nan,
                    "mean_return": np.nan,
                    "volatility": np.nan,
                    "sharpe_like": -np.inf
                })
                continue

            sub["strategy_return"] = sub["trade_direction_base"] * sub["future_return"]

            mean_ret = sub["strategy_return"].mean()
            vol = sub["strategy_return"].std()
            hit_rate = (sub["strategy_return"] > 0).mean()
            sharpe_like = mean_ret / vol if pd.notnull(vol) and vol > 0 else -np.inf

            rows.append({
                "threshold": thr,
                "n_trades": len(sub),
                "hit_rate": hit_rate,
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe_like": sharpe_like
            })

        threshold_df = pd.DataFrame(rows).sort_values(
            ["sharpe_like", "mean_return", "hit_rate"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        best_threshold = float(threshold_df.iloc[0]["threshold"]) if len(threshold_df) > 0 else float(threshold_grid[0])

        return best_threshold, threshold_df

    def apply_meta_filter(
        self,
        meta_df: pd.DataFrame,
        meta_prob,
        chosen_threshold: float
    ) -> pd.DataFrame:
        out = meta_df.copy()
        out["meta_probability"] = meta_prob

        out["selected_trade"] = (
            (out["predicted_class"] != 1) &
            (out["meta_probability"] >= chosen_threshold)
        ).astype(int)

        out["trade_direction"] = np.where(
            out["selected_trade"] == 1,
            out["trade_direction_base"],
            0
        )

        out["strategy_return"] = out["trade_direction"] * out["future_return"]

        return out

    def build_prediction_output(
        self,
        df: pd.DataFrame,
        model_name: str,
        chosen_threshold: float
    ) -> pd.DataFrame:
        out = df.copy()

        keep_cols = [
            "Date",
            "Ticker",
            "Close",
            "future_close",
            "future_return",
            "target_class",
            "naive_prediction_class",
            "predicted_class",
            "predicted_probability",
            "pred_prob_down",
            "pred_prob_flat",
            "pred_prob_up",
            "meta_probability",
            "selected_trade",
            "trade_direction",
            "strategy_return",
            "is_correct",
        ]

        out["model_name"] = model_name
        out["chosen_threshold"] = chosen_threshold

        return out[keep_cols + ["model_name", "chosen_threshold"]].copy()

    def feature_importance_table(
        self,
        model: XGBClassifier,
        feature_cols: List[str],
        ticker: str
    ) -> pd.DataFrame:
        importance_df = pd.DataFrame({
            "Ticker": ticker,
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df