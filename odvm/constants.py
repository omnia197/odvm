from enum import Enum
from typing import Dict, Any

class ProblemType(str, Enum):
    """
    Enumeration of supported machine learning problem types.

    Attributes:
        CLASSIFICATION: Classification problems (discrete target).
        REGRESSION: Regression problems (continuous target).
        CLUSTERING: Clustering problems (unsupervised).
        TIMESERIES: Time series forecasting or analysis.
    """
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    CLUSTERING = 'clustering'
    TIMESERIES = 'timeseries'

class DataType(str, Enum):
    """
    Enumeration of data types for feature characterization.

    Attributes:
        NUMERICAL: Numeric features (int, float).
        CATEGORICAL: Categorical/discrete features.
        TEXT: Textual/string features.
        DATETIME: Date and time features.
        BOOLEAN: Boolean features (True/False).
        TIMEDELTA: Time duration features.
    """
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    TEXT = 'text'
    DATETIME = 'datetime'
    BOOLEAN = 'boolean'
    TIMEDELTA = 'timedelta'

DEFAULT_CONFIG: Dict[str, Any] = {
    'preprocessing': {
        'numeric_strategy': 'median',
        'categorical_strategy': 'constant',
        'scale_numeric': True,
        'text_vectorization': 'tfidf'
    },
    'visualization': {
        'sample_size': 1000,
        'max_features': 20,
        'correlation_threshold': 0.8
    },
    'model_training': {
        'test_size': 0.2,
        'random_state': 42,
        'time_budget': 60
    }
}
"""
DEFAULT_CONFIG (dict): Default configuration for preprocessing, visualization, and model training.

Sections:
    preprocessing: Default strategies for numeric, categorical, and text features.
    visualization: Default sample size, feature limits, and correlation threshold.
    model_training: Default test size, random state, and time budget for training.
"""

DEFAULT_SAMPLE_SIZE = 1000
"""int: Default sample size for visualization and profiling."""

MAX_FEATURES_FOR_VISUALIZATION = 20
"""int: Maximum number of features to include in visualizations."""