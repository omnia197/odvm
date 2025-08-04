import pandas as pd
from sklearn.model_selection import train_test_split
from utils.detect_backend import detect_backend

class DataSplitter:
    """
    Splits a dataset into training, testing, and optionally validation sets.

    Supports both Pandas and Dask (after computing).

    Parameters
    ----------
    df : pd.DataFrame or dask.DataFrame
        The full dataset including features and target.
    target : str
        The name of the target column.
    task_type : str, optional
        The type of ML task. Used to decide stratification.
        Common values: 'classification', 'regression'.
    config : dict, optional
        Dictionary containing split configuration. Keys:
        - test_size : float (default=0.2)
            Proportion of data to use for testing.
        - val_size : float (default=0.0)
            Proportion of data to use for validation.
        - random_state : int (default=42)
            Random seed for reproducibility.
    """

    def __init__(self, df, target, task_type="classification", config=None):
        self.df = df
        self.target = target
        self.task_type = task_type
        self.config = config or {}
        self.backend = detect_backend(df)

    def split(self):
        """
        Splits the data into train/test or train/val/test sets.

        Returns
        -------
        X_train : pd.DataFrame
            Training features.
        X_val : pd.DataFrame or None
            Validation features (if val_size > 0).
        X_test : pd.DataFrame
            Testing features.
        y_train : pd.Series
            Training target.
        y_val : pd.Series or None
            Validation target (if val_size > 0).
        y_test : pd.Series
            Testing target.
        """

        test_size = self.config.get("test_size", 0.2)
        val_size = self.config.get("val_size", 0.0)  #default: no validation
        random_state = self.config.get("random_state", 42)

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        stratify = y if self.task_type.startswith("class") else None

        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )

            if val_size > 0:
                #make val_size relative to remaining data
                relative_val_size = val_size / (1.0 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=relative_val_size,
                    random_state=random_state,
                    stratify=stratify
                )
                print(f"Data split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                print(f"Data split: {len(X_temp)} train / {len(X_test)} test")
                return X_temp, None, X_test, y_temp, None, y_test

        except Exception as e:
            print(f"Split failed: {e}")
            return None, None, None, None, None, None
