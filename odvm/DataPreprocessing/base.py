from abc import ABC, abstractmethod
from typing import Union, List
import pandas as pd
import dask.dataframe as dd

from DataProfiler.base import DataProfile
from ..exceptions import  DataValidationError

class BaseDataPreprocessor(ABC):
    """
    Abstract base class for data preprocessors.

    This class defines the interface and common validation logic for all data preprocessor implementations.
    Subclasses must implement the `preprocess` method to perform data preprocessing based on the provided data profile.

    Methods:
        preprocess(data, profile):
            Abstract method to preprocess data based on its profile. Must be implemented by subclasses.

        _validate_input_data(data, profile):
            Validates the input data and profile before preprocessing. Raises DataValidationError if validation fails.

        _get_features_by_type(profile, data_type):
            Returns a list of feature names from the profile that match the specified data type.

    Raises:
        DataPreprocessingError: For errors during preprocessing.
        DataValidationError: For validation errors in the input data or profile.
    """

    def __init__(self):
        self.preprocessor = None

    @abstractmethod
    def preprocess(self, data: Union[pd.DataFrame, dd.DataFrame], 
                  profile: 'DataProfile') -> Union[pd.DataFrame, dd.DataFrame]:
        """Preprocess data based on its profile."""
        pass

    def _validate_input_data(self, data: Union[pd.DataFrame, dd.DataFrame], 
                           profile: 'DataProfile') -> None:
        """Validate input data before preprocessing.

        Checks that the DataFrame is not empty, has columns, and that all columns in the profile exist in the data.

        Args:
            data (Union[pd.DataFrame, dd.DataFrame]): The input data to validate.
            profile (DataProfile): The data profile containing feature information.

        Raises:
            DataValidationError: If the data is empty, has no columns, or columns in the profile are missing from the data.
        """
        if len(data) == 0:
            raise DataValidationError("DataFrame is empty")
        if len(data.columns) == 0:
            raise DataValidationError("DataFrame has no columns")
        for col, dtype in profile.feature_types.items():
            if col not in data.columns:
                raise DataValidationError(f"Column '{col}' in profile not found in data")

    def _get_features_by_type(self, profile: 'DataProfile', data_type: str) -> List[str]:
        """Get features of a specific type from the profile.

        Args:
            profile (DataProfile): The data profile containing feature information.
            data_type (str): The data type to filter features by.

        Returns:
            List[str]: A list of feature names matching the specified data type.
        """
        return [col for col, dtype in profile.feature_types.items() if dtype == data_type]