from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pickle
import logging
import pandas as pd
import dask.dataframe as dd

from DataLoading.dask_loader import DaskDataLoader
from DataLoading.pandas_loader import PandasDataLoader
from DataPreprocessing.categorical import CategoricalPreprocessor
from DataPreprocessing.numeric import NumericPreprocessor
from DataPreprocessing.text import TextPreprocessor
from DataProfiler.base import DataProfile
from DataProfiler.dask_profiler import DaskDataProfiler
from DataProfiler.pandas_profiler import PandasDataProfiler
from DataVisualization.distributions import DistributionVisualizer
from DataVisualization.missing_data import MissingDataVisualizer
from DataVisualization.relationships import RelationshipVisualizer
from ModelExplanation.feature_importance import FeatureImportanceExplainer
from ModelExplanation.shap_explainer import ShapExplainer
from ModelTrainer.classifiers import ClassifierTrainer
from ModelTrainer.regressors import RegressorTrainer
from .constants import DEFAULT_CONFIG, DataType, ProblemType
from .exceptions import (ODVMError, SerializationError, DataPreprocessingError)

logger = logging.getLogger(__name__)

class ODVM:
    """
    Main pipeline class for Organized Data Visualization & Machine (ODVM).

    This class provides an end-to-end interface for data loading, profiling, preprocessing,
    model training, explanation, and visualization. It supports both pandas and dask backends,
    and is configurable via a config file or dictionary.

    Args:
        config_path (Optional[Union[str, Path]]): Path to a configuration file. If None, uses default config.
        use_dask (bool): Whether to use Dask for scalable data processing. Default is False.
        random_state (int): Random seed for reproducibility. Default is 42.

    Attributes:
        use_dask (bool): Indicates if Dask is used for data processing.
        random_state (int): The random seed used for all components.
        config (dict): Pipeline configuration.
        data_profile (Optional[DataProfile]): The current data profile.
        _processed_data (Optional[Union[pd.DataFrame, dd.DataFrame]]): Preprocessed data.
        models (Dict[str, Any]): Dictionary of trained models.
        results (Dict[str, Any]): Dictionary of training and evaluation results.
        loader: Data loader instance (Dask or Pandas).
        profiler: Data profiler instance (Dask or Pandas).
        numeric_preprocessor: Numeric feature preprocessor.
        categorical_preprocessor: Categorical feature preprocessor.
        text_preprocessor: Text feature preprocessor.
        trainer: Model trainer instance (set after profiling).
        shap_explainer: SHAP model explainer.
        feature_importance_explainer: Feature importance explainer.
        distribution_visualizer: Distribution visualizer.
        relationship_visualizer: Relationship visualizer.
        missing_data_visualizer: Missing data visualizer.

    Methods:
        load(file_path, **kwargs):
            Load data from supported file formats.

        profile(data, target_col):
            Generate a comprehensive data profile.

        preprocess(data):
            Preprocess data based on its profile.

        train(data, target_col, time_budget):
            Train and evaluate machine learning models.

        explain(data, target_col):
            Explain model predictions using SHAP and feature importance.

        visualize(data, plot_type):
            Generate data visualizations.

        save(file_path):
            Save the complete pipeline state to disk.

        load(file_path):
            Load a saved pipeline state from disk.

        _get_features_by_type(data_type):
            Get features of a specific type from the profile.

    Raises:
        ValueError: If random_state is not a positive integer.
        ODVMError: For general pipeline errors.
        SerializationError: If saving or loading the pipeline fails.
        DataPreprocessingError: If preprocessing is attempted before profiling.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None,
                 use_dask: bool = False, random_state: int = 42):
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a positive integer")

        self.use_dask = use_dask
        self.random_state = random_state
        self.config = self._load_config(config_path)
        self.data_profile: Optional[DataProfile] = None
        self._processed_data: Optional[Union[pd.DataFrame, dd.DataFrame]] = None
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

        # Initialize components
        self.loader = DaskDataLoader() if use_dask else PandasDataLoader()
        self.profiler = DaskDataProfiler() if use_dask else PandasDataProfiler()
        self.numeric_preprocessor = NumericPreprocessor(
            strategy=self.config['preprocessing']['numeric_strategy'],
            scale=self.config['preprocessing']['scale_numeric']
        )
        self.categorical_preprocessor = CategoricalPreprocessor(
            strategy=self.config['preprocessing']['categorical_strategy']
        )
        self.text_preprocessor = TextPreprocessor(
            vectorization=self.config['preprocessing']['text_vectorization']
        )
        self.trainer = None  # Will be initialized based on problem type
        self.shap_explainer = ShapExplainer()
        self.feature_importance_explainer = FeatureImportanceExplainer()
        self.distribution_visualizer = DistributionVisualizer()
        self.relationship_visualizer = RelationshipVisualizer()
        self.missing_data_visualizer = MissingDataVisualizer()

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration from file."""
        from .utils import load_config
        return load_config(config_path, DEFAULT_CONFIG)

    def load(self, file_path: Union[str, Path], **kwargs) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load data from supported file formats."""
        self.data_profile = None
        self._processed_data = None
        return self.loader.load(file_path, **kwargs)

    def profile(self, data: Union[pd.DataFrame, dd.DataFrame],
               target_col: Optional[str] = None) -> DataProfile:
        """Generate comprehensive data profile."""
        self.data_profile = self.profiler.profile(data, target_col)
        return self.data_profile

    def preprocess(self, data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame]:
        """Preprocess data based on its profile."""
        if self.data_profile is None:
            raise DataPreprocessingError("Data profile is required before preprocessing. Call .profile() first.")

        numeric_features = self._get_features_by_type(DataType.NUMERICAL)
        categorical_features = self._get_features_by_type(DataType.CATEGORICAL)
        text_features = self._get_features_by_type(DataType.TEXT)

        processed_numeric = self.numeric_preprocessor.preprocess(data, numeric_features)
        processed_categorical = self.categorical_preprocessor.preprocess(data, categorical_features)
        processed_text = self.text_preprocessor.preprocess(data, text_features)

        # Combine processed features (implementation depends on your needs)
        # This is a simplified version - you may need to handle column names, etc.
        self._processed_data = pd.concat([processed_numeric, processed_categorical, processed_text], axis=1)
        return self._processed_data

    def train(self, data: Union[pd.DataFrame, dd.DataFrame],
             target_col: str,
             time_budget: int = 60) -> Dict[str, Any]:
        """Train and evaluate machine learning models."""
        if self.data_profile is None or self.data_profile.target_column != target_col:
            self.profile(data, target_col)

        # Initialize appropriate trainer
        if self.data_profile.problem_type == ProblemType.CLASSIFICATION:
            self.trainer = ClassifierTrainer(self.random_state)
        elif self.data_profile.problem_type == ProblemType.REGRESSION:
            self.trainer = RegressorTrainer(self.random_state)
        else:
            raise ODVMError(f"Unsupported problem type: {self.data_profile.problem_type}")

        # Use or generate processed data for training
        if self._processed_data is None:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            X_processed = self.preprocess(X)
        else:
            if target_col in self._processed_data.columns:
                X_processed = self._processed_data.drop(columns=[target_col])
                y = self._processed_data[target_col]
            else:
                X_processed = self._processed_data
                y = data[target_col]

        results = self.trainer.train(X_processed, y, time_budget)
        self.models = {'best': self.trainer.best_model}
        self.results['training'] = results
        return results

    def explain(self, data: Union[pd.DataFrame, dd.DataFrame],
               target_col: Optional[str] = None):
        """Explain model predictions using SHAP and feature importance."""
        if 'best' not in self.models or self.models['best'] is None:
            raise ODVMError("No trained model available for explanation")

        if self.data_profile is None:
            self.profile(data, target_col)

        if self._processed_data is None:
            X = data if target_col is None else data.drop(columns=[target_col])
            X_processed = self.preprocess(X)
        else:
            X_processed = self._processed_data.drop(columns=[self.data_profile.target_column])

        self.shap_explainer.explain(self.models['best'], X_processed)
        self.feature_importance_explainer.explain(self.models['best'], X_processed)

    def visualize(self, data: Union[pd.DataFrame, dd.DataFrame],
                plot_type: Optional[str] = None):
        """Generate data visualizations."""
        if self.data_profile is None:
            self.profile(data)

        if plot_type == 'distribution' and self.data_profile.target_column:
            self.distribution_visualizer.visualize(data, self.data_profile, self.data_profile.target_column)
        elif plot_type == 'relationships':
            self.relationship_visualizer.visualize(data, self.data_profile)
        elif plot_type == 'missing':
            self.missing_data_visualizer.visualize(data)
        else:
            # Default visualization
            if self.data_profile.target_column:
                self.distribution_visualizer.visualize(data, self.data_profile, self.data_profile.target_column)
            else:
                self.relationship_visualizer.visualize(data, self.data_profile)

    def save(self, file_path: Union[str, Path]):
        """Save the complete pipeline state to disk."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Pipeline successfully saved to {file_path}")
        except Exception as e:
            raise SerializationError(f"Failed to save pipeline: {str(e)}") from e

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'ODVM':
        """Load a saved pipeline state from disk."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise SerializationError(f"File not found: {file_path}")
            if file_path.stat().st_size == 0:
                raise SerializationError("File is empty")

            with open(file_path, 'rb') as f:
                return pickle.load(f)

        except pickle.UnpicklingError as e:
            raise SerializationError(f"Invalid pickle file: {str(e)}") from e
        except Exception as e:
            raise SerializationError(f"Failed to load pipeline: {str(e)}") from e

    def _get_features_by_type(self, data_type: str) -> List[str]:
        """Get features of a specific type from the profile."""
        if self.data_profile is None:
            raise DataPreprocessingError("Data profile is required. Call .profile() first.")
        return [col for col, dtype in self.data_profile.feature_types.items() if dtype == data_type]