from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    IsolationForest
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

#optional imports for advanced models
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None


class ModelSelector:
    """
    ModelSelector

    Selects appropriate models based on the task type.
    Supports classification, regression, clustering, anomaly detection, and dimensionality reduction.

    Parameters:
    ----------
    task_type : str
        Type of ML task (e.g., 'classification', 'regression', etc.)
    config : dict, optional
        Configuration specifying allowed/excluded models.
    """

    def __init__(self, task_type="classification", config=None):
        self.task_type = task_type
        self.config = config or {}

    def get_models(self):
        """
        Returns a dictionary of models appropriate for the specified task type.

        Returns:
        --------
        dict
            Dictionary of model instances.
        """
        if self.task_type in ["classification", "binary_classification"]:
            return self._classification_models()
        elif self.task_type == "regression":
            return self._regression_models()
        elif self.task_type == "clustering":
            return self._clustering_models()
        elif self.task_type == "anomaly_detection":
            return self._anomaly_models()
        elif self.task_type == "dimensionality_reduction":
            return self._reduction_models()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _classification_models(self):
        """
        Returns default classification models.

        Returns:
        --------
        dict
            Dictionary of classification model instances.
        """
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "NaiveBayes": GaussianNB()
        }

        if XGBClassifier:
            models["XGBoostClassifier"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        if LGBMClassifier:
            models["LightGBMClassifier"] = LGBMClassifier()
        if CatBoostClassifier:
            models["CatBoostClassifier"] = CatBoostClassifier(verbose=0)

        return self._filter(models)

    def _regression_models(self):
        """
        Returns default regression models.

        Returns:
        --------
        dict
            Dictionary of regression model instances.
        """
        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SVR": SVR()
        }

        if XGBRegressor:
            models["XGBoostRegressor"] = XGBRegressor()
        if LGBMRegressor:
            models["LightGBMRegressor"] = LGBMRegressor()
        if CatBoostRegressor:
            models["CatBoostRegressor"] = CatBoostRegressor(verbose=0)

        return self._filter(models)

    def _clustering_models(self):
        """
        Returns clustering models.

        Returns:
        --------
        dict
            Dictionary of clustering model instances.
        """
        models = {
            "KMeans": KMeans(n_clusters=3),
            "DBSCAN": DBSCAN()
        }
        return self._filter(models)

    def _anomaly_models(self):
        """
        Returns anomaly detection models.

        Returns:
        --------
        dict
            Dictionary of anomaly detection model instances.
        """
        models = {
            "IsolationForest": IsolationForest(),
            "OneClassSVM": OneClassSVM()
        }
        return self._filter(models)

    def _reduction_models(self):
        """
        Returns dimensionality reduction models.

        Returns:
        --------
        dict
            Dictionary of dimensionality reduction model instances.
        """
        models = {
            "PCA": PCA(n_components=2)
        }
        return self._filter(models)

    def _filter(self, models_dict):
        """
        Filters models based on allowed/excluded lists in config.

        Parameters:
        -----------
        models_dict : dict
            Dictionary of model instances before filtering.

        Returns:
        --------
        dict
            Filtered dictionary of models.
        """
        allowed = self.config.get("allowed_models")
        excluded = self.config.get("excluded_models")

        if allowed:
            models_dict = {k: v for k, v in models_dict.items() if k in allowed}
        if excluded:
            models_dict = {k: v for k, v in models_dict.items() if k not in excluded}

        print(f"Selected models: {list(models_dict.keys())}")
        return models_dict
