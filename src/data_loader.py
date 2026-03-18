# src/data_loader.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import yfinance as yf


@dataclass
class MarketDataLoader:
    symbols: List[str]
    start_date: str

    def _sanitize_column_name(self, name: str) -> str:
        """
        Make column names safe for Spark/Delta tables.
        """
        name = name.replace(" ", "_")
        name = name.replace("^", "")
        name = name.replace("/", "_")
        name = name.replace("-", "_")
        name = name.replace("(", "")
        name = name.replace(")", "")
        name = name.replace(",", "")
        name = name.replace("'", "")
        name = name.replace("\n", "_")
        name = name.replace("\t", "_")
        return name

    def download_data(self) -> pd.DataFrame:
        """
        Download OHLCV data for all requested symbols from Yahoo Finance.
        Returns a flattened pandas DataFrame with a Date column.
        """
        raw = yf.download(
            self.symbols,
            start=self.start_date,
            auto_adjust=False,
            progress=False
        )

        if raw.empty:
            raise ValueError("Downloaded dataframe is empty.")

        raw.columns = [
            "_".join([str(x) for x in col if x]).strip()
            for col in raw.columns
        ]

        raw.columns = [self._sanitize_column_name(col) for col in raw.columns]

        raw = raw.reset_index()
        raw.columns = [self._sanitize_column_name(col) for col in raw.columns]

        raw = raw.sort_values("Date").reset_index(drop=True)

        return raw

    def validate_download(self, df: pd.DataFrame) -> None:
        """
        Basic validation to make sure download looks correct.
        """
        if "Date" not in df.columns:
            raise ValueError("Expected 'Date' column was not found.")

        if len(df) == 0:
            raise ValueError("Downloaded dataframe has zero rows.")

    def save_to_table(self, spark, df: pd.DataFrame, table_name: str) -> None:
        """
        Save pandas dataframe to a Databricks/Spark table.
        """
        spark_df = spark.createDataFrame(df)
        spark_df.write.mode("overwrite").saveAsTable(table_name)