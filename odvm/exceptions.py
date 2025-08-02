class ODVMError(Exception):
    """
    Base exception class for all ODVM-specific errors.

    All custom exceptions in the ODVM package should inherit from this class.
    """
    pass

class DataLoadingError(ODVMError):
    """
    Exception raised for data loading failures.

    Raised when data cannot be loaded due to file issues, unsupported formats, or read errors.
    """
    pass

class DataValidationError(ODVMError):
    """
    Exception raised for data validation issues.

    Raised when input data fails validation checks, such as missing columns or invalid types.
    """
    pass

class DataProfilingError(ODVMError):
    """
    Exception raised during data profiling.

    Raised when profiling a dataset fails due to unexpected errors or invalid data.
    """
    pass

class DataPreprocessingError(ODVMError):
    """
    Exception raised during data preprocessing.

    Raised when preprocessing steps fail, such as imputation, encoding, or transformation errors.
    """
    pass

class ModelTrainingError(ODVMError):
    """
    Exception raised during model training.

    Raised when model training fails due to invalid data, convergence issues, or resource constraints.
    """
    pass

class ModelEvaluationError(ODVMError):
    """
    Exception raised during model evaluation.

    Raised when evaluation metrics cannot be computed or evaluation fails.
    """
    pass

class ModelInterpretationError(ODVMError):
    """
    Exception raised during model interpretation.

    Raised when model explanation or interpretation fails, such as SHAP or feature importance errors.
    """
    pass

class VisualizationError(ODVMError):
    """
    Exception raised during visualization.

    Raised when visualization generation fails due to plotting errors or invalid data.
    """
    pass

class ConfigurationError(ODVMError):
    """
    Exception raised for configuration issues.

    Raised when there are problems with user or system configuration.
    """
    pass

class SerializationError(ODVMError):
    """
    Exception raised for serialization/deserialization issues.

    Raised when saving or loading objects fails due to format or compatibility problems.
    """
    pass