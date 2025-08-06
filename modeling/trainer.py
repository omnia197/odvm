class ModelTrainer:
    """
    ModelTrainer

    This class handles training of multiple models on the provided dataset.

    It supports:
    - Supervised models (using both X and y)
    - Unsupervised models (using only X)
    - Transformer models (e.g., dimensionality reduction)

    Parameters:
    ----------
    models_dict : dict
        Dictionary of model name → model instance.
    X_train : DataFrame
        Features for training.
    y_train : Series or None
        Target labels. Can be None for unsupervised learning.
    """

    def __init__(self, models_dict, X_train, y_train=None):
        self.models = models_dict
        self.X_train = X_train
        self.y_train = y_train
        self.trained_models = {}
        self.trained_params = {}

    def train_all(self):
        """
        Train all models in the dictionary.

        For supervised models, uses X_train and y_train.
        For unsupervised models (e.g., clustering, dimensionality reduction), uses only X_train.

        Returns:
        --------
        trained_models : dict
            Dictionary of model name → trained model.
        trained_params : dict
            Dictionary of model name → selected/basic parameters.
        """
        for name, model in self.models.items():
            try:
                if self.y_train is not None:
                    model.fit(self.X_train, self.y_train)
                else:
                    model.fit(self.X_train)

                self.trained_models[name] = model

                if hasattr(model, "best_params_"):
                    self.trained_params[name] = model.best_params_
                    print(f"Best parameters for {name}: {model.best_params_}")
                else:
                    important_keys = ["n_estimators", "max_depth", "learning_rate", "n_clusters", "n_components"]
                    all_params = model.get_params()
                    basic = {k: all_params[k] for k in important_keys if k in all_params}
                    self.trained_params[name] = basic

                print(f"Trained: {name}")

            except Exception as e:
                print(f"Failed to train {name}: {e}")

        return self.trained_models, self.trained_params

    def get_trained(self):
        """
        Returns the dictionary of trained models.

        Returns:
        --------
        dict
            Trained models.
        """
        return self.trained_models
