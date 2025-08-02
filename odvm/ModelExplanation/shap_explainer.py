import shap
from typing import Union
import numpy as np
import pandas as pd
from ..exceptions import ModelInterpretationError

class ShapExplainer:
    """
    Provides model explanations using SHAP values.

    This class generates SHAP value explanations for machine learning models. It supports both tree-based models
    (using TreeExplainer) and other model types (using KernelExplainer). The explanation is visualized using SHAP's
    summary plots. Both pandas DataFrames and NumPy arrays are supported as input for feature data.

    Methods:
        explain(model, X):
            Generates and visualizes SHAP value explanations for the provided model and input data.

    Raises:
        ModelInterpretationError: If SHAP explanation fails due to errors in model compatibility or computation.
    """
    
    def explain(self, model, X: Union[pd.DataFrame, np.ndarray]):
        """Generate SHAP values explanation.

        Args:
            model: A trained model object to be explained.
            X (Union[pd.DataFrame, np.ndarray]): Feature data used for explanation.

        Raises:
            ModelInterpretationError: If SHAP explanation fails.
        """
        try:
            if hasattr(model, 'predict_proba'):
                predict_func = model.predict_proba
            else:
                predict_func = model.predict

            if str(type(model)).lower().find('ensemble') != -1 or \
               str(type(model)).lower().find('tree') != -1:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values, X, plot_type="bar")
                else:
                    shap.summary_plot(shap_values, X)
            else:
                explainer = shap.KernelExplainer(predict_func, shap.sample(X, 100))
                shap_values = explainer.shap_values(X)
                shap.summary_plot(shap_values, X)
        except Exception as e:
            raise ModelInterpretationError(f"SHAP explanation failed: {str(e)}") from e