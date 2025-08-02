from pathlib import Path
from typing import Union, Dict, Any
import pandas as pd
from .base import BaseDataLoader
from ..exceptions import DataLoadingError

class PandasDataLoader(BaseDataLoader):
    """
    Data loader implementation utilizing pandas for reading a wide range of file formats.

    This loader supports reading CSV, Excel, Parquet, JSON, Feather, and HDF files using pandas.
    The loader performs file validation prior to loading and raises a DataLoadingError if loading fails
    or if the file format is unsupported.

    Methods:
        load(file_path, **kwargs):
            Loads data from the specified file path using pandas.

    Raises:
        DataLoadingError: If the file cannot be loaded, is empty, corrupt, or in an unsupported format.
    """
    
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data using pandas."""
        try:
            file_path = Path(file_path)
            self._validate_file(file_path)
            suffix = file_path.suffix.lower()
            
            if suffix in self.SUPPORTED_FORMATS['csv']:
                return pd.read_csv(file_path, **kwargs)
            elif suffix in self.SUPPORTED_FORMATS['excel']:
                return pd.read_excel(file_path, **kwargs)
            elif suffix in self.SUPPORTED_FORMATS['parquet']:
                return pd.read_parquet(file_path, **kwargs)
            elif suffix in self.SUPPORTED_FORMATS['json']:
                return pd.read_json(file_path, **kwargs)
            elif suffix in self.SUPPORTED_FORMATS['feather']:
                return pd.read_feather(file_path, **kwargs)
            elif suffix in self.SUPPORTED_FORMATS['hdf']:
                return pd.read_hdf(file_path, **kwargs)
                
            raise DataLoadingError(f"Unsupported file format: {suffix}")
        except pd.errors.EmptyDataError:
            raise DataLoadingError(f"File is empty or corrupt: {file_path}")
        except Exception as e:
            raise DataLoadingError(f"Pandas loading failed for {file_path}: {str(e)}") from e