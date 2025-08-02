from pathlib import Path
from typing import Union, Dict, Any
import dask.dataframe as dd


from .base import BaseDataLoader
from ..exceptions import DataLoadingError

class DaskDataLoader(BaseDataLoader):
    """
    Data loader implementation utilizing Dask for efficient handling of large datasets.

    This loader supports reading CSV, Parquet, and JSON files using Dask. For file formats not natively supported by Dask,
    it falls back to using the PandasDataLoader. The loader performs file validation prior to loading and raises a
    DataLoadingError if loading fails.

    Methods:
        load(file_path, **kwargs):
            Loads data from the specified file path using Dask, or falls back to Pandas for unsupported formats.

    Raises:
        DataLoadingError: If the file cannot be loaded or is in an unsupported format.
    """

    def load(self, file_path: Union[str, Path], **kwargs) -> dd.DataFrame:
        """Load data using dask."""
        try:
            file_path = Path(file_path)
            self._validate_file(file_path)
            suffix = file_path.suffix.lower()
            
            if suffix in self.SUPPORTED_FORMATS['csv']:
                return dd.read_csv(file_path, **kwargs)
            elif suffix in self.SUPPORTED_FORMATS['parquet']:
                return dd.read_parquet(file_path, **kwargs)
            elif suffix in self.SUPPORTED_FORMATS['json']:
                return dd.read_json(file_path, **kwargs)
            else:
                # Fallback to pandas for formats not supported by dask
                from DataLoading.pandas_loader import PandasDataLoader
                return PandasDataLoader().load(file_path, **kwargs)
                
        except Exception as e:
            raise DataLoadingError(f"Dask loading failed for {file_path}: {str(e)}") from e