import plotly.graph_objects as go
import plotly.express as px
from typing import Union, List
import pandas as pd
import dask.dataframe as dd
import numpy as np

from DataProfiler.base import DataProfile
from ..constants import DataType
from .base import BaseDataVisualizer
from ..exceptions import VisualizationError

class RelationshipVisualizer(BaseDataVisualizer):
    """
    Visualizes relationships between features.

    This class provides methods to visualize relationships between numerical features,
    including correlation matrices and pairwise scatter plots. It supports both pandas
    and dask DataFrames. For large datasets, a sample is taken for visualization.

    Methods:
        visualize(data, profile, max_features):
            Visualizes feature relationships using a correlation heatmap and, for a small
            number of features, a scatter matrix plot.

    Raises:
        VisualizationError: If visualization fails due to errors in data or plotting.
    """
    
    def visualize(self, data: Union[pd.DataFrame, dd.DataFrame],
                profile: 'DataProfile',
                max_features: int = 20):
        """Visualize feature relationships.

        Args:
            data (Union[pd.DataFrame, dd.DataFrame]): The input data to visualize.
            profile (DataProfile): The data profile containing feature types.
            max_features (int): Maximum number of numerical features to include in the correlation matrix.

        Raises:
            VisualizationError: If visualization fails.
        """
        try:
            sample = data.sample(1000).compute() if isinstance(data, dd.DataFrame) else data.sample(1000)
            
            # Correlation matrix
            num_cols = [col for col, dtype in profile.feature_types.items() if dtype == DataType.NUMERICAL]
            if len(num_cols) >= 2:
                if len(num_cols) > max_features:
                    num_cols = num_cols[:max_features]
                
                corr = sample[num_cols].corr()
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1
                    )
                )
                fig.update_layout(title="Feature Correlation Matrix", width=800, height=800)
                fig.show()
                
            # Pairplot for small number of features
            if len(num_cols) >= 2 and len(num_cols) <= 5:
                fig = px.scatter_matrix(sample[num_cols])
                fig.update_traces(diagonal_visible=False)
                fig.update_layout(title="Numerical Features Pairplot")
                fig.show()
        except Exception as e:
            raise VisualizationError(f"Relationship visualization failed: {str(e)}") from e