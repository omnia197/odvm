from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)

class ModelEvaluator:
    """
    Evaluates a set of trained machine learning models based on the type of task.

    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained model instances with model names as keys.
    trained_params : dict
        Dictionary of training parameters used for each model.
    X_test : array-like or DataFrame
        Feature set used for testing.
    y_test : array-like or Series, optional
        True labels for testing (required for supervised tasks).
    task_type : str
        Type of machine learning task. One of:
        - 'classification' or 'binary_classification'
        - 'regression'
        - 'clustering'
        - 'dimensionality_reduction'
    """

    def __init__(self, trained_models, trained_params, X_test, y_test=None, task_type="classification"):
        self.models = trained_models
        self.params = trained_params
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type

    def evaluate(self):
        """
        Evaluate all models using task-appropriate metrics.

        Returns:
        --------
        results : list of dict
            A list containing evaluation results per model, including scores
            such as accuracy, F1-score, R2, or silhouette score depending on the task type.
        """
        results = []

        for name, model in self.models.items():
            try:
                if self.task_type in ["classification", "binary_classification"]:
                    y_pred = model.predict(self.X_test)
                    acc = accuracy_score(self.y_test, y_pred)
                    f1 = f1_score(self.y_test, y_pred, average='weighted')
                    prec = precision_score(self.y_test, y_pred, average='weighted')
                    rec = recall_score(self.y_test, y_pred, average='weighted')

                    results.append({
                        "Model": name,
                        "Params": self.params.get(name, {}),
                        "Accuracy": round(acc, 4),
                        "F1 Score": round(f1, 4),
                        "Precision": round(prec, 4),
                        "Recall": round(rec, 4),
                    })

                elif self.task_type == "regression":
                    y_pred = model.predict(self.X_test)
                    mse = mean_squared_error(self.y_test, y_pred)
                    mae = mean_absolute_error(self.y_test, y_pred)
                    r2 = r2_score(self.y_test, y_pred)

                    results.append({
                        "Model": name,
                        "Params": self.params.get(name, {}),
                        "MSE": round(mse, 4),
                        "MAE": round(mae, 4),
                        "R2 Score": round(r2, 4),
                        "Score": round(r2, 4)
                    })

                elif self.task_type == "clustering":
                    labels = model.fit_predict(self.X_test)
                    score = silhouette_score(self.X_test, labels)

                    results.append({
                        "Model": name,
                        "Params": self.params.get(name, {}),
                        "Silhouette Score": round(score, 4),
                        "Score": round(score, 4)
                    })

                elif self.task_type == "dimensionality_reduction":
                    results.append({
                        "Model": name,
                        "Params": self.params.get(name, {}),
                        "Score": "N/A (dimensionality reduction)"
                    })

                else:
                    print(f"[WARNING] Unknown task type: {self.task_type}")

            except Exception as e:
                print(f"[ERROR] Evaluation failed for {name}: {e}")

        return results
