from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from ..exceptions import ModelTrainingError, DataValidationError

class BaseModelTrainer(ABC):
    """
    Abstract base class for model trainers.

    This class defines the interface and common logic for all model trainer implementations.
    It manages model storage, training results, and tracks the best model and score.
    Subclasses must implement the `train` method to perform model training and evaluation.

    Args:
        random_state (int): Random seed for reproducibility. Default is 42.

    Attributes:
        random_state (int): The random seed used for model training.
        models (Dict[str, Any]): Dictionary of trained models.
        results (Dict[str, Any]): Dictionary of training results and metrics.
        best_model (Optional[Any]): The best performing model after training.
        best_score (float): The best score achieved during training.
        best_model_name (Optional[str]): Name of the best performing model.

    Methods:
        train(X, y, time_budget):
            Abstract method to train and evaluate models. Must be implemented by subclasses.

        _validate_input_data(X, y):
            Validates the input features and target before training.

    Raises:
        ValueError: If random_state is not a positive integer.
        DataValidationError: If input data is invalid.
    """
    
    def __init__(self, random_state: int = 42):
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a positive integer")
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_score: float = -np.inf
        self.best_model_name: Optional[str] = None
    
    @abstractmethod
    def train(self, X, y, time_budget: int = 60) -> Dict[str, Any]:
        """Train and evaluate models.

        Args:
            X: Feature data for training.
            y: Target values for training.
            time_budget (int): Maximum time (in seconds) allowed for training. Default is 60.

        Returns:
            Dict[str, Any]: Training results and metrics.

        Raises:
            ModelTrainingError: If training fails.
        """
        pass
    
    def _validate_input_data(self, X, y) -> None:
        """Validate input data before training.

        Checks that X and y have the same number of samples and are not empty.

        Args:
            X: Feature data.
            y: Target values.

        Raises:
            DataValidationError: If X and y have different lengths or are empty.
        """
        if len(X) != len(y):
            raise DataValidationError("X and y must have the same number of samples")
        if len(X) == 0:
            raise DataValidationError("Input data is empty")