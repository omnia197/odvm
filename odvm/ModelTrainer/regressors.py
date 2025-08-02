from typing import Dict, Any
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)
from .base import BaseModelTrainer
from ..exceptions import ModelTrainingError

class RegressorTrainer(BaseModelTrainer):
    """
    Handles training and evaluation of regression models.

    This class manages the training, evaluation, and selection of the best regression model
    from a set of supported algorithms. It supports Linear Regression, Random Forest, XGBoost,
    and LightGBM regressors. The class computes common regression metrics and tracks the best
    performing model based on R^2 score.

    Methods:
        train(X, y, time_budget):
            Trains and evaluates multiple regression models on the provided data, computes metrics,
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
        """Train and evaluate regression models.

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
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(random_state=self.random_state),
                'XGBoost': xgb.XGBRegressor(random_state=self.random_state),
                'LightGBM': lgb.LGBMRegressor(random_state=self.random_state)
            }
            
            metrics = {
                'mse': mean_squared_error,
                'mae': mean_absolute_error,
                'r2': r2_score,
                'explained_variance': explained_variance_score
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
                    
                    result['score'] = result['r2']
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
            raise ModelTrainingError(f"Regression training failed: {str(e)}") from e