from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import pandas as pd
import dask.dataframe as dd
from ..exceptions import DataLoadingError

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders in the package.
    This class defines the interface and common validation logic for all data loader implementations.
    It enforces the implementation of the `load` method for loading data from various supported file formats
    into pandas or dask DataFrames. The class also provides a file validation utility to ensure the integrity
    and existence of input files before loading.
    Attributes:
        SUPPORTED_FORMATS (dict): Mapping of format names to lists of supported file extensions.
    Methods:
        load(file_path, **kwargs):
            Abstract method to load data from the specified file path. Must be implemented by subclasses.
        _validate_file(file_path):
            Validates the existence, type, and non-emptiness of the input file.
    """
    """Abstract base class for data loaders."""
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv', '.tsv'],
        'excel': ['.xlsx', '.xls', '.xlsm'],
        'parquet': ['.parquet', '.pq'],
        'json': ['.json', '.jsonl'],
        'feather': ['.feather'],
        'hdf': ['.h5', '.hdf5']
    }

    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Abstract method to load data from the specified file path.

        Args:
            file_path (Union[str, Path]): The path to the data file to be loaded.
            **kwargs: Additional keyword arguments specific to the loader implementation.

        Returns:
            Union[pd.DataFrame, dd.DataFrame]: Loaded data as a pandas or dask DataFrame.

        Raises:
            DataLoadingError: If the file cannot be loaded or is in an unsupported format.
        """
        pass

    def _validate_file(self, file_path: Path) -> None:
        """
        Validate the input file before loading.

        Checks that the file exists, is a file (not a directory), and is not empty.

        Args:
            file_path (Path): The path to the file to validate.

        Raises:
            DataLoadingError: If the file does not exist, is not a file, or is empty.
        """
        if not file_path.exists():
            raise DataLoadingError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise DataLoadingError(f"Path is not a file: {file_path}")
        if file_path.stat().st_size == 0:
            raise DataLoadingError(f"File is empty: {file_path}")