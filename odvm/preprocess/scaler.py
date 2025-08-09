from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.detect_backend import detect_backend

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


class Scaler:
    """
    Applies feature scaling to numeric columns using a specified strategy.

    Supports both Pandas and Dask DataFrames.

    Parameters
    ----------
    strategy : str
        Scaling strategy to use. Options:
        - 'none': No scaling (default).
        - 'standard': Standardization (mean = 0, std = 1).
        - 'minmax': Normalization to [0, 1] range.
    """

    def __init__(self, strategy="none"):
        """
        Initializes the Scaler with the desired scaling strategy.

        Parameters
        ----------
        strategy : str
            Chosen scaling strategy.
        """
        self.strategy = strategy.lower()
        self.scaler = None
        self.backend = None

        if self.strategy == "standard":
            self.scaler = StandardScaler()
        elif self.strategy == "minmax":
            self.scaler = MinMaxScaler()

    def fit_transform(self, X):
        """
        Fits the scaler to the training data and applies the transformation.

        Parameters
        ----------
        X : pd.DataFrame or dask.DataFrame
            The feature matrix.

        Returns
        -------
        pd.DataFrame or dask.DataFrame
            Scaled features.
        """
        self.backend = detect_backend(X)

        if self.scaler:
            if self.backend == "pandas":
                return self._fit_transform_pandas(X)
            elif self.backend == "dask":
                return self._fit_transform_dask(X)

        return X

    def transform(self, X):
        """
        Applies the transformation to new (test/validation) data.

        Parameters
        ----------
        X : pd.DataFrame or dask.DataFrame
            The feature matrix.

        Returns
        -------
        pd.DataFrame or dask.DataFrame
            Scaled features.
        """
        if not self.scaler:
            return X

        if self.backend == "pandas":
            return self._transform_pandas(X)
        elif self.backend == "dask":
            return self._transform_dask(X)

        return X

    def _fit_transform_pandas(self, X):
        """
        Internal: Fit and transform using pandas.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        numeric_cols = X.select_dtypes(include=["int", "float"]).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        return X_scaled

    def _transform_pandas(self, X):
        """
        Internal: Transform only using pandas.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        numeric_cols = X.select_dtypes(include=["int", "float"]).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X_scaled

    def _fit_transform_dask(self, X):
        """
        Internal: Fit and transform using dask with sampling.

        Parameters
        ----------
        X : dask.DataFrame

        Returns
        -------
        dask.DataFrame
        """
        sample = X.sample(frac=0.2).compute()
        numeric_cols = sample.select_dtypes(include=["int", "float"]).columns
        self.scaler.fit(sample[numeric_cols])

        def scale_partition(partition):
            partition[numeric_cols] = self.scaler.transform(partition[numeric_cols])
            return partition

        return X.map_partitions(scale_partition)

    def _transform_dask(self, X):
        """
        Internal: Transform only using dask.

        Parameters
        ----------
        X : dask.DataFrame

        Returns
        -------
        dask.DataFrame
        """
        numeric_cols = X.select_dtypes(include=["int", "float"]).columns

        def scale_partition(partition):
            partition[numeric_cols] = self.scaler.transform(partition[numeric_cols])
            return partition

        return X.map_partitions(scale_partition)
