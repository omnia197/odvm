import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

from ..utils.detect_backend import detect_backend


class EDAVisualizer:
    """
    A class for visualizing exploratory data analysis (EDA) results.
    Supports both pandas and dask DataFrames.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame (pandas or dask).
    target : str, optional
        The target column to highlight or analyze.
    max_samples : int, optional
        The maximum number of samples to visualize (default is 5000).
    save_dir : str, optional
        Directory to save plots as PNG files (if provided).
    """

    def __init__(self, df, target=None, max_samples=5000, save_dir=None):
        self.raw_df = df
        self.target = target
        self.backend = detect_backend(df)
        self.df = self._prepare_df(max_samples)
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def _save_fig(self, name):
        """
        Save the current matplotlib figure to disk if save_dir is provided.

        Parameters
        ----------
        name : str
            Name of the file (without extension).
        """
        if self.save_dir:
            path = os.path.join(self.save_dir, f"{name}.png")
            plt.savefig(path)
            print(f"Saved figure: {path}")

    def _prepare_df(self, max_samples):
        """
        Prepare and sample the DataFrame for plotting.

        Parameters
        ----------
        max_samples : int
            Number of maximum samples to use for visualization.

        Returns
        -------
        DataFrame
            A sample or full DataFrame for plotting.
        """
        if self.backend == "pandas":
            df = self.raw_df.copy()
            if len(df) > max_samples:
                return df.sample(n=max_samples, random_state=42)
            return df
        elif self.backend == "dask":
            return self.raw_df.sample(frac=1.0).head(max_samples).compute()

    def plot_distributions(self, cols=None):
        """
        Plot histogram or countplot for selected columns.

        Parameters
        ----------
        cols : list, optional
            List of columns to visualize. If None, numerical and categorical columns will be included.
        """
        cols = cols or self.df.select_dtypes(include=["number", "object", "category"]).columns

        for col in cols:
            plt.figure(figsize=(6, 4))
            if self.df[col].dtype in ['object', 'category'] or self.df[col].nunique() < 15:
                sns.countplot(x=col, data=self.df)
                plt.title(f"Countplot of {col}")
            else:
                sns.histplot(self.df[col].dropna(), kde=True)
                plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            self._save_fig(f"distribution_{col}")
            # plt.show()

    def plot_boxplots(self, cols=None):
        """
        Plot boxplots for numeric columns.

        Parameters
        ----------
        cols : list, optional
            Columns to include. If None, all numeric columns are plotted.
        """
        df = self.df
        cols = cols or df.select_dtypes(include=["number"]).columns

        for col in cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(y=df[col].dropna())
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            self._save_fig(f"boxplot_{col}")
            # plt.show()

    def plot_correlation(self):
        """
        Plot correlation heatmap for numeric features.
        """
        df = self.df.select_dtypes(include=["number"])
        if df.shape[1] < 2:
            print("Not enough numerical features for correlation matrix.")
            return

        corr = df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        self._save_fig("correlation_heatmap")
        # plt.show()

    def plot_target_distribution(self):
        """
        Plot histogram of the target column, if defined.
        """
        if self.target and self.target in self.df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[self.target].dropna(), kde=True)
            plt.title(f"Distribution of Target: {self.target}")
            plt.tight_layout()
            self._save_fig(f"target_distribution_{self.target}")
            # plt.show()
        else:
            print("No valid target specified.")

    def plot_pairplot(self, max_features=5):
        """
        Plot pairwise scatterplots for numeric features and optionally target.

        Parameters
        ----------
        max_features : int, optional
            Maximum number of features to include (default is 5).
        """
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        selected = numeric_cols[:max_features]
        if self.target and self.target not in selected:
            selected.append(self.target)

        if len(selected) > 1:
            sns.pairplot(self.df[selected], hue=self.target if self.target in selected else None)
            plt.suptitle("Pairplot of Selected Features", y=1.02)
            self._save_fig("pairplot")
            # plt.show()
        else:
            print("Not enough features to generate pairplot.")
