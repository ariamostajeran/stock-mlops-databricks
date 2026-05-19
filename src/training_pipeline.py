# src/training_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd
import numpy as np


@dataclass
class TrainingPipeline:
    builder: Any
    trainer: Any
    top_k_features: int
    rolling_train_years: int
    rolling_test_months: int
    news_date_limits: Dict[str, tuple]

    def run_research_training(
        self,
        ml_df: pd.DataFrame,
        tickers: List[str]
    ):
        all_metrics = []
        all_predictions = []
        all_feature_importance = []
        all_probability_summary = []
        all_feature_selection = []
        all_best_params = []

        for ticker in tickers:
            ticker_feature_cols = self.builder.get_feature_columns(ml_df, ticker=ticker)

            ticker_df, model_feature_cols = self.trainer.prepare_single_ticker_dataset(
                ml_df=ml_df,
                feature_cols=ticker_feature_cols,
                ticker=ticker
            )

            ticker_df = ticker_df.dropna(
                subset=model_feature_cols + ["target_up_down", "naive_prediction"]
            ).reset_index(drop=True)

            ticker_df["Date"] = pd.to_datetime(ticker_df["Date"], errors="coerce")

            if ticker in self.news_date_limits:
                start_date, end_date = self.news_date_limits[ticker]
                ticker_df = ticker_df[
                    (ticker_df["Date"] >= pd.Timestamp(start_date)) &
                    (ticker_df["Date"] <= pd.Timestamp(end_date))
                ].copy().reset_index(drop=True)

            if len(ticker_df) == 0:
                continue

            splits = self.trainer.get_walk_forward_splits(
                ticker_df,
                train_years=self.rolling_train_years,
                test_months=self.rolling_test_months
            )

            if len(splits) == 0:
                continue

            for split_id, (train_df, test_df) in enumerate(splits, start=1):
                X_train_full = train_df[model_feature_cols].copy()
                y_train = train_df["target_up_down"].copy()

                X_test_full = test_df[model_feature_cols].copy()
                y_test = test_df["target_up_down"].copy()

                model_full = self.trainer.fit_xgboost(X_train_full, y_train)
                top_features = self.trainer.select_top_features(model_full, model_feature_cols, self.top_k_features)

                X_train = X_train_full[top_features].copy()
                X_test = X_test_full[top_features].copy()

                model, best_params, best_cv_score = self.trainer.tune_xgboost(X_train, y_train, n_splits=3)

                all_best_params.append({
                    "Ticker": ticker,
                    "split_id": split_id,
                    "best_cv_auc": best_cv_score,
                    **best_params
                })

                train_pred = model.predict(X_train)
                train_prob = model.predict_proba(X_train)[:, 1]

                test_pred = model.predict(X_test)
                test_prob = model.predict_proba(X_test)[:, 1]

                train_metrics = self.trainer.evaluate_predictions(y_train, train_pred, train_prob)
                test_metrics = self.trainer.evaluate_predictions(y_test, test_pred, test_prob)

                all_metrics.append({
                    "Ticker": ticker,
                    "split_id": split_id,
                    "dataset_split": "train",
                    **train_metrics
                })
                all_metrics.append({
                    "Ticker": ticker,
                    "split_id": split_id,
                    "dataset_split": "test",
                    **test_metrics
                })

                pred_df = self.trainer.build_prediction_output(
                    test_df, y_pred=test_pred, y_prob=test_prob, model_name=f"{ticker}_weekly_model"
                )
                pred_df["split_id"] = split_id
                all_predictions.append(pred_df)

                imp_df = self.trainer.feature_importance_table(model=model, feature_cols=top_features, ticker=ticker)
                imp_df["split_id"] = split_id
                all_feature_importance.append(imp_df)

                all_feature_selection.append({
                    "Ticker": ticker,
                    "split_id": split_id,
                    "selected_feature_count": len(top_features),
                    "selected_features": ", ".join(top_features)
                })

                test_conf = pd.Series(test_prob)
                all_probability_summary.append({
                    "Ticker": ticker,
                    "split_id": split_id,
                    "test_prob_q25": float(test_conf.quantile(0.25)),
                    "test_prob_median": float(test_conf.median()),
                    "test_prob_q75": float(test_conf.quantile(0.75)),
                    "test_prob_max": float(test_conf.max()),
                    "train_start_date": train_df["Date"].min(),
                    "train_end_date": train_df["Date"].max(),
                    "test_start_date": test_df["Date"].min(),
                    "test_end_date": test_df["Date"].max(),
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                })

        return {
            "metrics_df": pd.DataFrame(all_metrics),
            "predictions_df": pd.concat(all_predictions, axis=0, ignore_index=True) if all_predictions else pd.DataFrame(),
            "importance_df": pd.concat(all_feature_importance, axis=0, ignore_index=True) if all_feature_importance else pd.DataFrame(),
            "probability_summary_df": pd.DataFrame(all_probability_summary),
            "feature_selection_df": pd.DataFrame(all_feature_selection),
            "best_params_df": pd.DataFrame(all_best_params),
        }

    def fit_final_inference_models(
        self,
        ml_df: pd.DataFrame,
        tickers: List[str]
    ):
        final_models = {}
        final_feature_sets = {}
        final_metadata = []

        for ticker in tickers:
            ticker_feature_cols = self.builder.get_feature_columns(ml_df, ticker=ticker)

            ticker_df, model_feature_cols = self.trainer.prepare_single_ticker_dataset(
                ml_df=ml_df,
                feature_cols=ticker_feature_cols,
                ticker=ticker
            )

            ticker_df = ticker_df.dropna(
                subset=model_feature_cols + ["target_up_down", "naive_prediction"]
            ).reset_index(drop=True)

            ticker_df["Date"] = pd.to_datetime(ticker_df["Date"], errors="coerce")

            if ticker in self.news_date_limits:
                start_date, end_date = self.news_date_limits[ticker]
                ticker_df = ticker_df[
                    (ticker_df["Date"] >= pd.Timestamp(start_date)) &
                    (ticker_df["Date"] <= pd.Timestamp(end_date))
                ].copy().reset_index(drop=True)

            if len(ticker_df) == 0:
                continue

            X_full = ticker_df[model_feature_cols].copy()
            y_full = ticker_df["target_up_down"].copy()

            model_full = self.trainer.fit_xgboost(X_full, y_full)
            top_features = self.trainer.select_top_features(model_full, model_feature_cols, self.top_k_features)

            X_final = X_full[top_features].copy()
            final_model, best_params, best_cv_score = self.trainer.tune_xgboost(X_final, y_full, n_splits=3)

            final_models[ticker] = final_model
            final_feature_sets[ticker] = top_features

            final_metadata.append({
                "Ticker": ticker,
                "n_rows": len(ticker_df),
                "n_features": len(top_features),
                "best_cv_auc": best_cv_score,
                **best_params
            })

        return final_models, final_feature_sets, pd.DataFrame(final_metadata)