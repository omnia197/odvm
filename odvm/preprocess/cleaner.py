import pandas as pd
import numpy as np
import logging
from utils.detect_backend import detect_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    DataCleaner class for performing data cleaning operations such as:
    - Removing duplicates
    - Handling missing values using multiple strategies
    - Detecting and handling outliers via IQR or Z-score methods

    Supports both Pandas and Dask DataFrames.
    """

    def __init__(self, df, config=None):
        """
        Initializes the DataCleaner.

        Parameters
        ----------
        df : pd.DataFrame or dask.dataframe.DataFrame
            The input DataFrame to clean.
        config : dict, optional
            Configuration dictionary for cleaning:
            - drop_duplicates (bool)
            - missing_strategy (str)
            - fill_value (any)
            - fill_strategy (dict)
            - outliers (dict)
        """
        self.original_df = df.copy()
        self.df = df
        self.config = config or {}
        self.backend = detect_backend(df)

    def _compute(self, expr):
        """
        Executes computation for Dask or returns expression for Pandas.

        Parameters
        ----------
        expr : expression
            The expression to compute.

        Returns
        -------
        Any
            Computed result.
        """
        return expr if self.backend == "pandas" else expr.compute()

    def _map(self, func):
        """
        Applies function using map_partitions for Dask or directly for Pandas.

        Parameters
        ----------
        func : callable
            The function to apply.

        Returns
        -------
        pd.DataFrame or dask.dataframe.DataFrame
            Transformed DataFrame.
        """
        return func if self.backend == "pandas" else self.df.map_partitions(func)

    def remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame if enabled in config.
        """
        if self.config.get("drop_duplicates", True):
            before = self._compute(self.df.shape[0])
            self.df = self.df.drop_duplicates()
            after = self._compute(self.df.shape[0])
            logger.info(f"Removed {before - after} duplicate rows.")

    def handle_missing(self):
        """
        Handles missing values in the DataFrame.

        Supported strategies:
        - "drop"
        - "mean", "median", "mode", "min", "max"
        - constant values
        - column-specific strategies via "fill_strategy"
        """
        strategy_map = self.config.get("fill_strategy", {})
        default_strategy = self.config.get("missing_strategy", "drop")
        fill_value = self.config.get("fill_value", 0)

        if default_strategy == "drop":
            before = self._compute(self.df.shape[0])
            self.df = self.df.dropna()
            after = self._compute(self.df.shape[0])
            logger.info(f"Dropped {before - after} rows with missing values.")
            return

        for col in self.df.columns:
            if self._compute(self.df[col].isnull().sum()) == 0:
                continue

            strategy = strategy_map.get(col, default_strategy)

            try:
                if strategy == "mean":
                    value = self._compute(self.df[col].mean())
                elif strategy == "median":
                    value = self._compute(self.df[col].median())
                elif strategy == "mode":
                    value = self._compute(self.df[col].mode().iloc[0])
                elif strategy == "min":
                    value = self._compute(self.df[col].min())
                elif strategy == "max":
                    value = self._compute(self.df[col].max())
                else:
                    value = strategy

                self.df[col] = self.df[col].fillna(value)
                logger.info(f"'{col}': filled using strategy '{strategy}' with value {value}")
            except Exception as e:
                logger.warning(f"'{col}': fill failed with strategy '{strategy}' — {e}")

    def handle_outliers(self):
        """
        Handles outliers in numerical columns using either:
        - IQR method (default)
        - Z-score method if enabled in config

        Supported strategies:
        - "remove": drops rows containing outliers
        - "cap": clips values to bounds
        """
        outlier_config = self.config.get("outliers", {})
        method = outlier_config.get("strategy", "remove")
        use_zscore = outlier_config.get("use_zscore", False)
        threshold = outlier_config.get("zscore_threshold", 3)
        selected_columns = outlier_config.get("columns", [])

        if not selected_columns:
            selected_columns = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        for col in selected_columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found.")
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Column '{col}' is not numeric — skipping.")
                continue

            if use_zscore:
                mean = self._compute(self.df[col].mean())
                std = self._compute(self.df[col].std())
                z_scores = (self.df[col] - mean) / std
                mask = z_scores.abs() > threshold
            else:
                Q1 = self._compute(self.df[col].quantile(0.25))
                Q3 = self._compute(self.df[col].quantile(0.75))
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (self.df[col] < lower) | (self.df[col] > upper)

            outliers_count = self._compute(mask.sum())
            if outliers_count == 0:
                logger.info(f"No outliers in '{col}'")
                continue

            logger.info(f"Found {outliers_count} outliers in '{col}'")

            if method == "remove":
                self.df = self.df[~mask].copy()
                logger.info(f"Removed outliers in '{col}'")
            elif method == "cap":
                if use_zscore:
                    capped = self.df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)
                else:
                    capped = self.df[col].clip(lower=lower, upper=upper)
                self.df[col] = capped
                logger.info(f"Capped outliers in '{col}'")
            else:
                logger.warning(f"Unknown outlier strategy: '{method}'")

    def clean(self):
        """
        Runs the full cleaning pipeline:
        - Remove duplicates
        - Handle missing values
        - Handle outliers

        Returns
        -------
        pd.DataFrame or dask.dataframe.DataFrame
            The cleaned DataFrame.
        """
        self.remove_duplicates()
        self.handle_missing()
        self.handle_outliers()
        return self.df
