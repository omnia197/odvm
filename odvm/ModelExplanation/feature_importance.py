import pandas as pd
import plotly.express as px
from typing import Union
import numpy as np
from ..exceptions import ModelInterpretationError

class FeatureImportanceExplainer:
    """
    Provides feature importance explanations for machine learning models.

    This class generates feature importance plots for models that expose a `feature_importances_` attribute,
    such as tree-based models. It supports both pandas DataFrames and NumPy arrays as input for feature data.

    Methods:
        explain(model, X):
            Generates a horizontal bar plot of the top 20 most important features using Plotly.

    Raises:
        ModelInterpretationError: If the feature importance plot cannot be generated due to errors.
    """
    
    def explain(self, model, X: Union[pd.DataFrame, np.ndarray]):
        """Generate feature importance plot.

        Args:
            model: A trained model object with a `feature_importances_` attribute.
            X (Union[pd.DataFrame, np.ndarray]): Feature data used for model training or prediction.

        Raises:
            ModelInterpretationError: If the feature importance plot cannot be generated.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                features = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]

                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                fig = px.bar(
                    importance_df.head(20),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                fig.show()
        except Exception as e:
            raise ModelInterpretationError(f"Feature importance plot failed: {str(e)}") from e