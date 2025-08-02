import plotly.express as px
from typing import Optional, Union
import pandas as pd
import dask.dataframe as dd

from DataProfiler.base import DataProfile
from ..constants import DataType
from .base import BaseDataVisualizer
from ..exceptions import VisualizationError

class DistributionVisualizer(BaseDataVisualizer):
    """
    Visualizes distributions of features.

    This class provides methods to visualize the distribution of a specified target column
    using Plotly. It supports both numerical and categorical features and works with both
    pandas and dask DataFrames. For large datasets, a sample is taken for visualization.

    Methods:
        visualize(data, profile, target_col):
            Visualizes the distribution of the specified target column using appropriate
            plot types (histogram, bar, etc.) based on the feature type.

    Raises:
        VisualizationError: If visualization fails due to errors in data or plotting.
    """
    
    def visualize(self, data: Union[pd.DataFrame, dd.DataFrame],
                profile: 'DataProfile',
                target_col: Optional[str] = None):
        """Visualize distributions.

        Args:
            data (Union[pd.DataFrame, dd.DataFrame]): The input data to visualize.
            profile (DataProfile): The data profile containing feature types.
            target_col (Optional[str]): The column to visualize the distribution for.

        Raises:
            VisualizationError: If visualization fails.
        """
        try:
            sample = data.sample(1000).compute() if isinstance(data, dd.DataFrame) else data.sample(1000)
            
            if target_col:
                target_type = profile.feature_types[target_col]
                if target_type == DataType.NUMERICAL:
                    if sample[target_col].nunique() > 10:
                        fig = px.histogram(sample, x=target_col, marginal="box", nbins=50)
                    else:
                        counts = sample[target_col].value_counts().reset_index()
                        fig = px.bar(counts, x=target_col, y='count')
                else:
                    counts = sample[target_col].value_counts().reset_index()
                    fig = px.bar(counts, y=target_col, x='count', orientation='h')
                fig.update_layout(title=f"Distribution of {target_col}")
                fig.show()
        except Exception as e:
            raise VisualizationError(f"Distribution visualization failed: {str(e)}") from e