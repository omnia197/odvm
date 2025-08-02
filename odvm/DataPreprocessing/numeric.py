from typing import List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .base import BaseDataPreprocessor
from ..exceptions import DataPreprocessingError

class NumericPreprocessor(BaseDataPreprocessor):
    """
    Preprocessor for handling numerical features.

    This class provides preprocessing for numerical data, including imputation of missing values
    and optional feature scaling. It uses a scikit-learn Pipeline with SimpleImputer and StandardScaler.
    The imputation strategy and scaling can be configured.

    Args:
        strategy (str): The imputation strategy for missing values. Default is 'median'.
        scale (bool): Whether to apply standard scaling to the features. Default is True.

    Methods:
        preprocess(data, numeric_features):
            Preprocesses the specified numerical features in the input data using the configured pipeline.

    Raises:
        DataPreprocessingError: If preprocessing fails.
    """
    
    def __init__(self, strategy: str = 'median', scale: bool = True):
        super().__init__()
        self.strategy = strategy
        self.scale = scale
    
    def preprocess(self, data, numeric_features: List[str]):
        """Preprocess numerical features.

        Applies imputation and optional scaling to the specified numerical features.

        Args:
            data: The input DataFrame containing the numerical features.
            numeric_features (List[str]): List of numerical feature names to preprocess.

        Returns:
            The processed numerical features as a NumPy array or DataFrame, depending on the pipeline.

        Raises:
            DataPreprocessingError: If preprocessing fails.
        """
        try:
            if not numeric_features:
                return data
            
            steps = [('imputer', SimpleImputer(strategy=self.strategy))]
            if self.scale:
                steps.append(('scaler', StandardScaler()))
                
            self.preprocessor = Pipeline(steps)
            processed = self.preprocessor.fit_transform(data[numeric_features])
            
            return processed
        except Exception as e:
            raise DataPreprocessingError(f"Numeric preprocessing failed: {str(e)}") from e