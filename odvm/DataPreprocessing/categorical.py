from typing import List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from .base import BaseDataPreprocessor
from ..exceptions import DataPreprocessingError

class CategoricalPreprocessor(BaseDataPreprocessor):
    """
    Preprocessor for handling categorical features.

    This class provides preprocessing for categorical data, including imputation of missing values
    and one-hot encoding. It uses a scikit-learn Pipeline with SimpleImputer and OneHotEncoder.
    The imputation strategy and handling of unknown categories can be configured.

    Args:
        strategy (str): The imputation strategy for missing values. Default is 'constant'.
        handle_unknown (str): How to handle unknown categories in OneHotEncoder. Default is 'ignore'.

    Methods:
        preprocess(data, categorical_features):
            Preprocesses the specified categorical features in the input data using the configured pipeline.

    Raises:
        DataPreprocessingError: If preprocessing fails.
    """
    
    def __init__(self, strategy: str = 'constant', handle_unknown: str = 'ignore'):
        super().__init__()
        self.strategy = strategy
        self.handle_unknown = handle_unknown
    
    def preprocess(self, data, categorical_features: List[str]):
        """Preprocess categorical features.

        Applies imputation and one-hot encoding to the specified categorical features.

        Args:
            data: The input DataFrame containing the categorical features.
            categorical_features (List[str]): List of categorical feature names to preprocess.

        Returns:
            The processed categorical features as a NumPy array or DataFrame, depending on the pipeline.

        Raises:
            DataPreprocessingError: If preprocessing fails.
        """
        try:
            if not categorical_features:
                return data
                
            self.preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy=self.strategy, fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False))
            ])
            processed = self.preprocessor.fit_transform(data[categorical_features])
            
            return processed
        except Exception as e:
            raise DataPreprocessingError(f"Categorical preprocessing failed: {str(e)}") from e