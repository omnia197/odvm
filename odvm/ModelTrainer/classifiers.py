from typing import Dict, Any
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from .base import BaseModelTrainer
from ..exceptions import ModelTrainingError

class ClassifierTrainer(BaseModelTrainer):
    """
    Handles training and evaluation of classification models.

    This class manages the training, evaluation, and selection of the best classification model
    from a set of supported algorithms. It supports Logistic Regression, Random Forest, XGBoost,
    and LightGBM classifiers. The class computes common classification metrics and tracks the best
    performing model based on accuracy.

    Methods:
        train(X, y, time_budget):
            Trains and evaluates multiple classification models on the provided data, computes metrics,
            and selects the best model.

    Attributes (inherited):
        random_state (int): The random seed used for model training.
        models (Dict[str, Any]): Dictionary of trained models.
        results (Dict[str, Any]): Dictionary of training results and metrics.
        best_model (Optional[Any]): The best performing model after training.
        best_score (float): The best score achieved during training.
        best_model_name (Optional[str]): Name of the best performing model.

    Raises:
        ModelTrainingError: If training or evaluation fails.
    """
    
    def train(self, X, y, time_budget: int = 60) -> Dict[str, Any]:
        """Train and evaluate classification models.

        Args:
            X: Feature data for training.
            y: Target values for training.
            time_budget (int): Maximum time (in seconds) allowed for training. Default is 60.

        Returns:
            Dict[str, Any]: Training results and metrics for each model.

        Raises:
            ModelTrainingError: If training or evaluation fails.
        """
        try:
            self._validate_input_data(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.random_state),
                'RandomForest': RandomForestClassifier(random_state=self.random_state),
                'XGBoost': xgb.XGBClassifier(random_state=self.random_state),
                'LightGBM': lgb.LGBMClassifier(random_state=self.random_state)
            }
            
            metrics = {
                'accuracy': accuracy_score,
                'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
                'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
                'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
                'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else None
            }
            
            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    result = {}
                    for metric_name, metric_func in metrics.items():
                        try:
                            result[metric_name] = metric_func(y_test, y_pred)
                        except Exception:
                            result[metric_name] = None
                    
                    result['score'] = result['accuracy']
                    results[name] = result
                    
                    if result['score'] > self.best_score:
                        self.best_score = result['score']
                        self.best_model = model
                        self.best_model_name = name
                        
                except Exception as e:
                    results[name] = {'error': str(e), 'status': 'failed'}
            
            self.results = results
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Classification training failed: {str(e)}") from e