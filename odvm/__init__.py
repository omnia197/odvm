"""
ODVM - Organized Data Visualization & Machine

A comprehensive pipeline for automated data analysis, visualization, and machine learning.
"""

from .constants import ProblemType, DataType
from .exceptions import (
    ODVMError, DataLoadingError, DataProfilingError, ModelTrainingError,
    DataValidationError, DataPreprocessingError, ModelEvaluationError,
    ModelInterpretationError, VisualizationError, ConfigurationError,
    SerializationError
)
from .odvm import ODVM

__version__ = "0.1.0"
__all__ = [
    'ODVM',
    'ProblemType',
    'DataType',
    'ODVMError',
    'DataLoadingError',
    'DataProfilingError',
    'ModelTrainingError',
    'DataValidationError',
    'DataPreprocessingError',
    'ModelEvaluationError',
    'ModelInterpretationError',
    'VisualizationError',
    'ConfigurationError',
    'SerializationError'
]