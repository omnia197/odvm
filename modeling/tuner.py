from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import numpy as np

class ModelTuner:
    """
    Tunes a model using grid or random search with optional cross-validation.

    Parameters
    ----------
    model : estimator
        The model to tune.
    param_grid : dict
        The hyperparameter grid.
    X_train : DataFrame
        Training features.
    y_train : Series
        Target labels.
    strategy : str, default="grid"
        Search strategy: "grid" or "random".
    cv : int, default=3
        Number of CV folds.
    scoring : str or callable, optional
        Scoring metric.
    n_iter : int, default=10
        Number of iterations for RandomSearch.
    cross_validate : bool, default=False
        Whether to run additional cross-validation on the best model.
    """

    def __init__(self, model, param_grid, X_train, y_train,
                 strategy="grid", cv=3, scoring=None, n_iter=10, cross_validate=False):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.strategy = strategy
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.cross_validate = cross_validate

    def tune(self):
        """
        Tunes the model using the selected search strategy and returns the best fitted model.

        Returns:
        --------
        best_model : estimator
            The best estimator found during search, already fitted on the training data.
        """
        print(f"[INFO] Tuning model: {type(self.model).__name__} using {self.strategy.upper()} search...")

        if self.strategy == "random":
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=1
            )

        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        print(f"[INFO] Best parameters for {type(self.model).__name__}: {best_params}")
        print(f"[INFO] Best CV Score: {round(best_score, 4)}")

        #extra cross-validation on best model
        if self.cross_validate:
            print(f"[INFO] Performing cross-validation on best model...")
            scores = cross_val_score(best_model, self.X_train, self.y_train, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(scores)
            print(f"[INFO] Cross-validated {self.scoring or 'score'}: {round(mean_score, 4)}")

        return best_model
