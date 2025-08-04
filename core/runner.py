# odvm/core/runner.py

from typing import Union, Optional
import pandas as pd

from assistant.task_detector import TaskDetector
from eda.visualizer import EDAVisualizer
from modeling.evaluator import ModelEvaluator
from modeling.explainer import ModelExplainer
from modeling.model_selector import ModelSelector
from modeling.trainer import ModelTrainer
from modeling.tuner import ModelTuner
from preprocess.cleaner import DataCleaner
from preprocess.encoder import Encoder
from preprocess.scaler import Scaler
from preprocess.splitter import DataSplitter
from reporting.report_builder import ReportBuilder
from utils.detect_backend import detect_backend
from eda.analyzer import EDAAnalyzer

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


class ODVM:
    """
    ODVM: Open, Dynamic, and Versatile Modeling

    This class controls the full AutoML pipeline:
    - EDA
    - Preprocessing
    - Modeling & Tuning
    - Reporting
    - Deployment (coming soon)
    """

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        target: str,
        backend: Optional[str] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize the ODVM pipeline.

        Parameters
        ----------
        data : str or DataFrame
            Input data path (CSV/Excel/Parquet) or loaded DataFrame.
        target : str
            Name of the target column.
        backend : str, optional
            Backend type: 'pandas', 'dask', or None for auto-detection.
        config : dict, optional
            Configuration for EDA, preprocessing, modeling, etc.
        """
        self.raw_data = self._load_data(data)
        self.target = target
        self.backend = backend or detect_backend(self.raw_data)
        self.config = config or {}
        self._validate()

        print(f"Data loaded | Shape: {self.raw_data.shape} | Backend: {self.backend}")

    def _load_data(self, data_input):
        """Load data from file path or accept existing DataFrame."""
        if isinstance(data_input, str):
            if data_input.endswith(".csv"):
                return pd.read_csv(data_input)
            elif data_input.endswith(".xlsx"):
                return pd.read_excel(data_input)
            elif dd and data_input.endswith(".parquet"):
                return dd.read_parquet(data_input)
            else:
                raise ValueError("Unsupported file format.")
        elif isinstance(data_input, pd.DataFrame):
            return data_input
        elif dd and isinstance(data_input, dd.DataFrame):
            return data_input
        else:
            raise TypeError("Unsupported data input type.")

    def _validate(self):
        """Validate the presence of the target column."""
        if self.target not in self.raw_data.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")

    def run(
        self,
        eda: bool = True,
        preprocess: bool = True,
        model: bool = True,
        report: bool = True,
        deploy: bool = False
    ):
        """
        Run the full pipeline.

        Parameters
        ----------
        eda : bool
            Whether to run exploratory data analysis.
        preprocess : bool
            Whether to run data cleaning, encoding, scaling, and splitting.
        model : bool
            Whether to run model selection, tuning, and training.
        report : bool
            Whether to generate a model performance report.
        deploy : bool
            Whether to deploy the model (API/dashboard).
        """
        task_type = TaskDetector(self.raw_data, self.target).detect()
        print(f"Detected task type: {task_type.upper()}")

        self.task_type = task_type

        if eda:
            print("\nRunning EDA...")
            self.run_eda()

        if preprocess:
            print("\nPreprocessing data...")
            self.run_preprocessing()

        if model:
            print("\nTraining model(s)...")
            self.run_modeling()

        if report:
            print("\nGenerating report...")
            self.generate_report()

        if deploy:
            print("\nDeploying model...")
            self.deploy_model()

    def run_eda(self):
        """Run EDA analysis and visualizations."""
        eda_config = self.config.get("eda", {})
        save_dir = eda_config.get("save_dir", None)

        print("Running EDA Analysis & Visualization...")

        analyzer = EDAAnalyzer(self.raw_data, target=self.target)
        print("Dataset shape:", analyzer.get_shape())
        print("Column Info:", analyzer.get_columns_info())
        print("Missing Values:\n", analyzer.missing_values())

        if eda_config.get("save_summary", True):
            summary_path = eda_config.get("summary_path", "outputs/eda_summary.json")
            analyzer.save_summary(path=summary_path)
            print(f"EDA summary saved to {summary_path}")

        if eda_config.get("visualize", True):
            visualizer = EDAVisualizer(self.raw_data, target=self.target, save_dir=save_dir)
            visualizer.plot_distributions()
            visualizer.plot_boxplots()
            visualizer.plot_correlation()
            visualizer.plot_target_distribution()
            visualizer.plot_pairplot()

    def run_preprocessing(self):
        """Run cleaning, encoding, splitting and scaling."""
        cleaner = DataCleaner(self.raw_data, self.config.get("preprocessing", {}))
        cleaned_df = cleaner.clean()

        encoder = Encoder(cleaned_df, self.config.get("preprocessing", {}))
        encoded_df = encoder.encode()

        splitter = DataSplitter(
            df=encoded_df,
            target=self.target,
            task_type=self.task_type,
            config=self.config.get("split", {})
        )
        X_train, X_test, y_train, y_test = splitter.split()

        scaling_strategy = self.config.get("preprocessing", {}).get("scaling", "none")
        scaler = Scaler(strategy=scaling_strategy)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test

        print("Data ready for modeling.")

    def run_modeling(self):
        """Run model selection, tuning (if enabled), training, and evaluation."""
        model_config = self.config.get("modeling", {})

        selector = ModelSelector(task_type=self.task_type, config=model_config)
        models = selector.get_models()

        tuned_models = {}
        tuning_enabled = model_config.get("tuning", False)
        strategy = model_config.get("tuning_strategy", "grid")
        param_grids = model_config.get("param_grids", {})
        scoring = model_config.get("scoring", None)

        for name, model in models.items():
            if tuning_enabled and name in param_grids:
                print(f"Tuning {name}...")
                tuner = ModelTuner(
                    model=model,
                    param_grid=param_grids[name],
                    X_train=self.X_train,
                    y_train=self.y_train,
                    strategy=strategy,
                    scoring=scoring
                )
                tuned_models[name] = tuner.tune()
            else:
                tuned_models[name] = model

        trainer = ModelTrainer(tuned_models, self.X_train, self.y_train)
        trained_models, trained_params = trainer.train_all()
        self.trained_models = trained_models

        evaluator = ModelEvaluator(trained_models, trained_params, self.X_test, self.y_test, self.task_type)
        results = evaluator.evaluate()
        self.results = results

        report = ReportBuilder(results)
        report.build()

    def generate_report(self):
        """Generate detailed evaluation and model performance reports."""
        pass  # To be implemented

    def deploy_model(self):
        """Deploy the trained model via API or dashboard."""
        pass  # To be implemented
