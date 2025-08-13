from ..utils.detect_backend import detect_backend
import os
import json

class EDAAnalyzer:
    """
    Perform automated exploratory data analysis (EDA) on a DataFrame (pandas or dask).

    Parameters
    ----------
    df : DataFrame
        The input DataFrame to analyze (can be pandas or dask).
    target : str, optional
        The name of the target column (default is None).

    Attributes
    ----------
    df : DataFrame
        The original input DataFrame.
    target : str
        Target column name.
    backend : str
        The backend type: 'pandas' or 'dask'.
    """

    def __init__(self, df, target=None):
        self.df = df
        self.target = target
        self.backend = detect_backend(df)

    def _compute(self, expr):
        """
        Evaluate expressions for dask or return directly for pandas.

        Parameters
        ----------
        expr : Expression
            An expression or DataFrame/Series to compute.

        Returns
        -------
        Computed value or the original object.
        """
        return expr if self.backend == "pandas" else expr.compute()

    def get_shape(self):
        """
        Get the shape of the DataFrame.

        Returns
        -------
        tuple
            (number of rows, number of columns)
        """
        return self._compute(self.df.shape)

    def get_columns_info(self):
        """
        Get data type, feature type, and number of unique values per column.

        Returns
        -------
        dict
            A dictionary with info per column.
        """
        info = {}
        for col in self.df.columns:
            col_data = self.df[col]
            dtype = str(col_data.dtype)
            if dtype.startswith("int") or dtype.startswith("float"):
                col_type = "numerical"
            elif "datetime" in dtype:
                col_type = "datetime"
            else:
                col_type = "categorical"
            info[col] = {
                "type": col_type,
                "dtype": dtype,
                "n_unique": self._compute(col_data.nunique())
            }
        return info

    def describe(self):
        """
        Get full statistical description of the dataset.

        Returns
        -------
        DataFrame
            Descriptive statistics for all columns.
        """
        return self._compute(self.df.describe(include="all"))

    def missing_values(self):
        """
        Count missing values per column.

        Returns
        -------
        Series
            Number of missing values per column.
        """
        return self._compute(self.df.isnull().sum())

    def constant_columns(self):
        """
        Find columns with only one unique value.

        Returns
        -------
        list
            List of constant column names.
        """
        return [col for col in self.df.columns if self._compute(self.df[col].nunique()) <= 1]

    def high_missing_columns(self, threshold=0.5):
        """
        Identify columns with a high ratio of missing values.

        Parameters
        ----------
        threshold : float, optional
            Proportion threshold to consider (default is 0.5).

        Returns
        -------
        list
            List of column names exceeding the missing threshold.
        """
        total_rows = self._compute(self.df.shape[0])
        nulls = self.missing_values()
        return [col for col, val in nulls.items() if val / total_rows > threshold]

    def top_frequent_values(self, top_n=3):
        """
        Get top N frequent values per column.

        Parameters
        ----------
        top_n : int, optional
            Number of top frequent values to return per column (default is 3).

        Returns
        -------
        dict
            Dictionary of top values per column.
        """
        result = {}
        for col in self.df.columns:
            try:
                value_counts = self._compute(self.df[col].value_counts().head(top_n))
                result[col] = value_counts.to_dict()
            except Exception:
                result[col] = {}
        return result

    def correlation_matrix(self):
        """
        Compute correlation matrix for numerical features.

        Returns
        -------
        DataFrame
            Correlation matrix.
        """
        num_cols = [col for col in self.df.columns if str(self.df[col].dtype).startswith(('int', 'float'))]
        df_numeric = self.df[num_cols]
        return self._compute(df_numeric.corr())

    def target_distribution(self):
        """
        Analyze the distribution of the target column.

        Returns
        -------
        str or Series or DataFrame
            Summary of the target distribution or error message.
        """
        if self.target is None or self.target not in self.df.columns:
            return "No target column found."
        target_col = self.df[self.target]
        if str(target_col.dtype).startswith(('int', 'float')) and self._compute(target_col.nunique()) > 20:
            return self._compute(target_col.describe())
        else:
            return self._compute(target_col.value_counts(normalize=True))

    def save_summary(self, path="outputs/eda_summary.json"):
        """
        Save full EDA summary as a JSON file.

        Parameters
        ----------
        path : str, optional
            File path to save the summary (default is 'outputs/eda_summary.json').

        Returns
        -------
        None
        """
        summary = {
            "shape": self.get_shape(),
            "columns_info": self.get_columns_info(),
            "missing_values": self.missing_values().to_dict(),
            "constant_columns": self.constant_columns(),
            "high_missing_columns": self.high_missing_columns(),
            "top_frequent_values": self.top_frequent_values(),
            "correlation_matrix": self.correlation_matrix().to_dict(),
            "target_distribution": self.target_distribution().to_dict()
            if not isinstance(self.target_distribution(), str)
            else str(self.target_distribution())
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"EDA summary saved to: {path}")
