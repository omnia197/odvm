import missingno as msno
import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
import dask.dataframe as dd
from .base import BaseDataVisualizer
from ..exceptions import VisualizationError

class MissingDataVisualizer(BaseDataVisualizer):
    """
    Visualizes missing data patterns in a dataset.

    This class provides methods to visualize the presence and distribution of missing values
    using the missingno library. It supports both pandas and dask DataFrames. For large datasets,
    a sample is taken for visualization.

    Methods:
        visualize(data):
            Visualizes missing data patterns using a matrix plot.

    Raises:
        VisualizationError: If visualization fails due to errors in data or plotting.
    """
    
    def visualize(self, data: Union[pd.DataFrame, dd.DataFrame]):
        """Visualize missing data patterns.

        Args:
            data (Union[pd.DataFrame, dd.DataFrame]): The input data to visualize for missing values.

        Raises:
            VisualizationError: If visualization fails.
        """
        try:
            sample = data.sample(1000).compute() if isinstance(data, dd.DataFrame) else data.sample(1000)
            msno.matrix(sample)
            plt.title("Missing Values Matrix")
            plt.show()
        except Exception as e:
            raise VisualizationError(f"Missing data visualization failed: {str(e)}") from e