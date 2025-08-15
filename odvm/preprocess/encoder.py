import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.detect_backend import detect_backend

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


class Encoder:
    """
    Automatically encodes categorical features using either Label or One-Hot encoding.

    Supports both Pandas and Dask DataFrames.

    Parameters
    ----------
    df : pd.DataFrame or dask.DataFrame
        Input dataframe containing categorical columns.
    config : dict, optional
        Configuration dictionary. Keys:
        - encoding: "label" or "onehot"
        - columns: List of columns to encode (default: all categoricals)
        - drop_first: bool, for one-hot encoding (default: True)
    """

    def __init__(self, df, config=None):
        """
        Initialize the Encoder with DataFrame and configuration.

        Parameters
        ----------
        df : pd.DataFrame or dask.DataFrame
            Input DataFrame.
        config : dict, optional
            Encoding configuration options.
        """
        self.df = df
        self.config = config or {}
        self.backend = detect_backend(df)
        self.encoding_type = self.config.get("encoding", "label").lower()
        self.columns = self.config.get("columns", None)
        self.drop_first = self.config.get("drop_first", True)

    def encode(self):
        """
        Applies the encoding to the specified columns.

        Returns
        -------
        pd.DataFrame or dask.DataFrame
            Encoded DataFrame.
        """
        if self.encoding_type == "label":
            return self._label_encode()
        elif self.encoding_type == "onehot":
            return self._one_hot_encode()
        else:
            print(f"[Encoder] Unknown encoding type: '{self.encoding_type}' — returning original DataFrame.")
            return self.df

    def _label_encode(self):
        """
        Applies label encoding to categorical columns.

        Returns
        -------
        pd.DataFrame or dask.DataFrame
            DataFrame with label-encoded columns.
        """
        df = self.df
        cols = self.columns or df.select_dtypes(include=["object", "category"]).columns

        if self.backend == "pandas":
            for col in cols:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                except Exception as e:
                    print(f"[⚠️] Skipping column '{col}' (Label encoding failed): {e}")
            return df

        elif self.backend == "dask":
            def label_encode_partition(partition, cols):
                for col in cols:
                    try:
                        le = LabelEncoder()
                        partition[col] = le.fit_transform(partition[col].astype(str))
                    except Exception:
                        pass
                return partition

            return df.map_partitions(label_encode_partition, cols)

    def _one_hot_encode(self):
        """
        Applies one-hot encoding to categorical columns.

        Returns
        -------
        pd.DataFrame or dask.DataFrame
            DataFrame with one-hot encoded columns.
        """
        df = self.df
        cols = self.columns or df.select_dtypes(include=["object", "category"]).columns

        if self.backend == "pandas":
            return pd.get_dummies(df, columns=cols, drop_first=self.drop_first)

        elif self.backend == "dask":
            try:
                return dd.get_dummies(df, columns=cols, drop_first=self.drop_first)
            except Exception as e:
                print(f"Dask one-hot encoding failed: {e}")
                return df
