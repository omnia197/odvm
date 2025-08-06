import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelExplainer:
    """
    Provides model interpretation using feature importance or SHAP values.

    Parameters
    ----------
    model : object
        A trained machine learning model. Should support either feature_importances_ or be compatible with SHAP.
    X_sample : DataFrame
        A sample of the input features used to explain the model.
    method : str, default="auto"
        Explanation method to use. Options:
        - "auto": Uses feature importance if available, otherwise falls back to SHAP if available.
        - "shap": Uses SHAP values for explanation (requires SHAP to be installed).
        - "importance": Uses the model's feature_importances_ attribute.
    """

    def __init__(self, model, X_sample, method="auto"):
        self.model = model
        self.X_sample = X_sample
        self.method = method

    def explain(self):
        """
        Runs the selected explanation method based on the specified configuration.
        """
        if self.method == "auto":
            if hasattr(self.model, "feature_importances_"):
                self.plot_feature_importance()
            elif SHAP_AVAILABLE:
                self.shap_summary()
            else:
                print("No explanation method available for this model.")
        elif self.method == "shap" and SHAP_AVAILABLE:
            self.shap_summary()
        elif self.method == "importance":
            self.plot_feature_importance()
        else:
            print("Invalid or unavailable explanation method.")

    def plot_feature_importance(self):
        """
        Plots feature importances based on the model's 'feature_importances_' attribute.
        """
        importances = self.model.feature_importances_
        features = self.X_sample.columns

        sorted_idx = importances.argsort()[::-1]
        sorted_features = features[sorted_idx]
        sorted_importances = importances[sorted_idx]

        plt.figure(figsize=(8, 5))
        plt.barh(sorted_features, sorted_importances)
        plt.gca().invert_yaxis()
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

    def shap_summary(self):
        """
        Generates a SHAP summary plot explaining the model's predictions.
        """
        explainer = shap.Explainer(self.model, self.X_sample)
        shap_values = explainer(self.X_sample)
        shap.summary_plot(shap_values, self.X_sample)
