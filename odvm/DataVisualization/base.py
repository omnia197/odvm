from abc import ABC, abstractmethod
from typing import Union, Optional
import pandas as pd
import dask.dataframe as dd

from DataProfiler.base import DataProfile
from ..exceptions import DataValidationError

class BaseDataVisualizer(ABC):
    """
    Abstract base class for data visualizers.

    This class defines the interface and common validation logic for all data visualizer implementations.
    Subclasses must implement the `visualize` method to generate visualizations based on the provided data
    and its profile.

    Methods:
        visualize(data, profile, plot_type):
            Abstract method to generate visualizations based on data characteristics and the specified plot type.

        _validate_input_data(data):
            Validates the input data before visualization. Raises DataValidationError if validation fails.

    Raises:
        DataValidationError: If the input data is empty or has no columns.
    """
    
    @abstractmethod
    def visualize(self, data: Union[pd.DataFrame, dd.DataFrame],
                profile: 'DataProfile',
                plot_type: Optional[str] = None):
        """Generate visualizations based on data characteristics and the specified plot type.

        Args:
            data (Union[pd.DataFrame, dd.DataFrame]): The input data to visualize.
            profile (DataProfile): The data profile containing metadata for visualization.
            plot_type (Optional[str]): The type of plot to generate (e.g., 'histogram', 'scatter').

        Raises:
            DataValidationError: If the input data is invalid.
        """
        pass
    
    def _validate_input_data(self, data: Union[pd.DataFrame, dd.DataFrame]) -> None:
        """Validate input data before visualization.

        Checks that the DataFrame is not empty and has columns.

        Args:
            data (Union[pd.DataFrame, dd.DataFrame]): The input data to validate.

        Raises:
            DataValidationError: If the DataFrame is empty or has no columns.
        """
        if len(data) == 0:
            raise DataValidationError("DataFrame is empty")
        if len(data.columns) == 0:
            raise DataValidationError("DataFrame has no columns")