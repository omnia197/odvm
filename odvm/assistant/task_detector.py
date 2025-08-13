import pandas as pd
from ..utils.detect_backend import detect_backend

class TaskDetector:
    def __init__(self, df, target = None):
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
        Detects the machine learning task type based on the target column.

        Returns
        -------
        dict:
            {
                "task_type": str,         # e.g. classification, regression, clustering, etc.
                "learning_type": str      # supervised or unsupervised
            }
        """
        if self.target is None or self.target not in self.sample.columns:
            #no target -> unsupervised machine learning
            return {
                "task_type": "clustering",
                "learning_type": "unsupervised"
            }
        target_series = self.sample[self.target]

        if "date" in str(self.sample.index.name).lower() or \
           pd.api.types.is_datetime64_any_dtype(target_series):
            return {
                "task_type": "time_series",
                "learning_type": "supervised"
            }

        unique_vals = target_series.nunique()
        if pd.api.types.is_numeric_dtype(target_series):
            if unique_vals <= 10:
                return {
                    "task_type": "binary_classification" if unique_vals == 2 else "classification",
                    "learning_type": "supervised"
                }
            else:
                return {
                    "task_type": "regression",
                    "learning_type": "supervised"
                }
        else:
            return {
                "task_type": "classification",
                "learning_type": "supervised"
            }

