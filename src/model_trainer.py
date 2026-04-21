# src/model_trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
            random_state=self.random_state,
            seed=self.random_state,
            n_estimators=250,
            max_depth=2,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=8,
            reg_alpha=3,
            min_child_weight=8,
            objective="binary:logistic",
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        return model
    def tune_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 3
    ):
        param_grid = {
            "n_estimators": [150, 250],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.02, 0.05],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.6, 0.8],
            "min_child_weight": [5, 8]
        }

        base_model = XGBClassifier(
            random_state=self.random_state,
            seed=self.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            reg_lambda=8,
            reg_alpha=3
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        return grid.best_estimator_, grid.best_params_, grid.best_score_
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
        y_pred,
        y_prob
    ) -> Dict[str, Any]:
        auc = roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else None

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "auc": auc,
        }

    def build_prediction_output(
        self,
        df: pd.DataFrame,
        y_pred,
        y_prob,
        model_name: str
    ) -> pd.DataFrame:
        out = df.copy()

        out["predicted_direction"] = y_pred
        out["predicted_probability"] = y_prob
        out["confidence"] = y_prob
        out["is_correct"] = (out["predicted_direction"] == out["target_up_down"]).astype(int)

        out["trade_direction"] = out["predicted_direction"].map({1: 1, 0: -1})
        out["strategy_return"] = out["trade_direction"] * out["future_return"]
        out["selected_trade"] = 1

        out["model_name"] = model_name

        keep_cols = [
            "Date",
            "Ticker",
            "Close",
            "future_close",
            "future_return",
            "target_up_down",
            "naive_prediction",
            "predicted_direction",
            "predicted_probability",
            "confidence",
            "selected_trade",
            "trade_direction",
            "strategy_return",
            "is_correct",
            "model_name",
        ]
        return out[keep_cols].copy()

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