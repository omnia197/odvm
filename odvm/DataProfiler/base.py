from abc import abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..exceptions import DataProfilingError, DataValidationError

from abc import abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..exceptions import DataProfilingError, DataValidationError

@dataclass
class DataProfile:
    """
    Stores a comprehensive profile of a dataset, including feature types, missing values,
    unique values, statistics, and metadata.

    Attributes:
        feature_types (Dict[str, str]): Mapping of feature names to their data types.
        missing_values (Dict[str, int]): Mapping of feature names to the count of missing values.
        unique_values (Dict[str, int]): Mapping of feature names to the number of unique values.
        stats (Dict[str, Dict[str, Any]]): Mapping of feature names to their statistical summaries.
        target_column (Optional[str]): Name of the target column, if applicable.
        problem_type (Optional[str]): Type of problem (e.g., classification, regression).
        dataset_shape (Optional[tuple]): Shape of the dataset as (rows, columns).
        memory_usage (Optional[int]): Memory usage of the dataset in bytes.
        column_metadata (Dict[str, Dict[str, Any]]): Additional metadata for each column.
    """
    feature_types: Dict[str, str] = None
    missing_values: Dict[str, int] = None
    unique_values: Dict[str, int] = None
    stats: Dict[str, Dict[str, Any]] = None
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    dataset_shape: Optional[tuple] = None
    memory_usage: Optional[int] = None
    column_metadata: Dict[str, Dict[str, Any]] = None

class BaseDataProfiler:
    """
    Abstract base class for data profilers.

    This class defines the interface and common validation logic for all data profiler implementations.
    Subclasses must implement the `profile` method to generate a DataProfile for a given dataset.

    Methods:
        profile(data, target_col):
            Abstract method to generate a comprehensive data profile for the input data.

        _validate_input_data(data):
            Validates the input data before profiling. Raises DataValidationError if validation fails.

    Raises:
        DataProfilingError: For errors during profiling.
        DataValidationError: For validation errors in the input data.
    """
    
    def __init__(self):
        pass
        
    @abstractmethod
    def profile(self, data, target_col: Optional[str] = None) -> DataProfile:
        """
        Generate a comprehensive data profile for the input dataset.

        Args:
            data: The input DataFrame to be profiled.
            target_col (Optional[str]): The name of the target column, if applicable.

        Returns:
            DataProfile: The generated data profile.

        Raises:
            DataProfilingError: If profiling fails.
        """
        pass
        
    def _validate_input_data(self, data) -> None:
        """
        Validate input data before profiling.

        Checks that the DataFrame is not empty and has columns.

        Args:
            data: The input DataFrame to validate.

        Raises:
            DataValidationError: If the DataFrame is empty or has no columns.
        """
        if len(data) == 0:
            raise DataValidationError("DataFrame is empty")
        if len(data.columns) == 0:
            raise DataValidationError("DataFrame has no columns")