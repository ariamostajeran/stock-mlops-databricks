# 📈 ML-Based Stock Trading Signal Framework

This project implements a machine learning pipeline for predicting short-term stock movements and generating trading signals.

## 🚀 Overview

The system combines:

- Feature engineering (technical + macro signals)
- Multi-class classification (UP / FLAT / DOWN)
- Per-stock XGBoost models
- Meta-labeling for trade filtering
- Threshold optimization
- Backtesting with trading metrics

## 🧠 Pipeline

Market Data → Features → Model → Meta Model → Trade Selection → Backtest

## 📊 Features

- Multi-horizon returns (1d → 60d)
- Moving averages & trend indicators
- Volatility measures
- Volume dynamics
- Cross-asset signals (SPY, QQQ, VIX, etc.)
- Regime features

## 🤖 Models

- Base model: XGBoost (multi-class)
- Meta model: Logistic Regression
- Feature selection: model-based (top-K features)

## 📈 Evaluation

- Accuracy, precision, recall
- Confidence-based filtering
- Rolling walk-forward validation
- Trading metrics:
  - hit rate
  - mean return
  - Sharpe-like ratio
  - cumulative return

## 🧪 Results

The system identifies tradable signals across multiple stocks with varying performance levels, demonstrating realistic alpha generation.

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Databricks

## 📌 Future Improvements

- Add news sentiment features
- Improve position sizing
- Portfolio optimization
- Transaction cost modeling