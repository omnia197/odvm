import pandas as pd
from utils.detect_backend import detect_backend

class TaskDetector:
    def __init__(self, df, target):
        """
        Initialize the TaskDetector.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the dataset.
        target : str
            The name of the target column.

        Attributes
        ----------
        df : pd.DataFrame
            Stores the full input DataFrame.
        target : str
            The name of the target column.
        backend : str
            Either 'pandas' or 'dask', based on input DataFrame.
        sample : pd.DataFrame
            A sample subset of the DataFrame (computed if Dask).
        """
        self.df = df
        self.target = target
        self.backend = detect_backend(df)
        self.sample = self._sample_df()

    def _sample_df(self):
        """
        Samples the input DataFrame if backend is Dask.

        Returns
        -------
        pd.DataFrame
            The full DataFrame if using Pandas, otherwise a sampled and computed subset of the Dask DataFrame.
        """
        if self.backend == "pandas":
            return self.df
        else:
            return self.df.sample(frac=0.3).compute()

    def detect(self):
        """
        Detects the machine learning task type (classification, regression, time series).

        Returns
        -------
        str
            The type of task detected:
            - "regression"
            - "binary_classification"
            - "classification"
            - "time_series"
        """
        target_series = self.sample[self.target]

        if "date" in str(self.sample.index.name).lower() or \
           pd.api.types.is_datetime64_any_dtype(target_series):
            return "time_series"

        unique_vals = target_series.nunique()
        if pd.api.types.is_numeric_dtype(target_series):
            if unique_vals <= 10:
                return "binary_classification" if unique_vals == 2 else "classification"
            else:
                return "regression"
        else:
            return "classification"
