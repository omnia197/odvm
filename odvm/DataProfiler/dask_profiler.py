import dask.dataframe as dd
import pandas as pd
from typing import Optional
from .base import BaseDataProfiler, DataProfile
from ..constants import DataType, ProblemType
from ..exceptions import DataProfilingError, DataValidationError

class DaskDataProfiler(BaseDataProfiler):
    """
    Data profiler implementation utilizing Dask for scalable analysis of large datasets.

    This class generates a comprehensive profile of a Dask DataFrame, including feature type detection,
    missing and unique value analysis, statistical summaries, and problem type inference. It is designed
    to efficiently handle large datasets by sampling when appropriate.

    Methods:
        profile(data, target_col):
            Generates a DataProfile for the provided Dask DataFrame, optionally using a specified target column.

    Raises:
        DataProfilingError: If profiling fails due to unexpected errors.
        DataValidationError: If validation of the input data or target column fails.
    """

    def profile(self, data: dd.DataFrame, target_col: Optional[str] = None) -> DataProfile:
        """Generate a comprehensive data profile for a Dask DataFrame.

        Args:
            data (dd.DataFrame): The input Dask DataFrame to be profiled.
            target_col (Optional[str]): The name of the target column, if applicable.

        Returns:
            DataProfile: The generated data profile containing feature types, statistics, and metadata.

        Raises:
            DataProfilingError: If profiling fails.
        """
        try:
            self._validate_input_data(data)
            profile = DataProfile()
            
            # Compute sample for feature type detection
            sample = data.sample(frac=0.1).compute() if len(data) > 1000 else data.compute()
            
            # Detect feature types
            self._detect_feature_types(sample, profile)
            
            # Calculate statistics on full data
            self._calculate_statistics(data, profile)
            
            # Detect problem type
            if target_col:
                self._validate_target_column(data, target_col)
                self._detect_problem_type(sample, profile, target_col)
            else:
                profile.problem_type = self._auto_detect_problem_type(profile, sample)
                
            return profile
        except Exception as e:
            raise DataProfilingError(f"Profiling failed: {str(e)}") from e

    def _validate_target_column(self, data: dd.DataFrame, target_col: str) -> None:
        """Validate that the target column exists and contains valid data.

        Args:
            data (dd.DataFrame): The input Dask DataFrame.
            target_col (str): The name of the target column.

        Raises:
            DataValidationError: If the target column is missing or contains only null values.
        """
        if target_col not in data.columns:
            raise DataValidationError(f"Target column '{target_col}' not found in data")
        if data[target_col].isnull().all().compute():
            raise DataValidationError(f"Target column '{target_col}' contains only null values")

    def _detect_feature_types(self, data: pd.DataFrame, profile: DataProfile):
        """Detect and record feature types for each column.

        Args:
            data (pd.DataFrame): A sample of the dataset as a pandas DataFrame.
            profile (DataProfile): The profile object to update.
        """
        profile.feature_types = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                profile.feature_types[col] = DataType.BOOLEAN if data[col].nunique() == 2 else DataType.NUMERICAL
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                profile.feature_types[col] = DataType.DATETIME
            elif pd.api.types.is_string_dtype(data[col]):
                profile.feature_types[col] = DataType.CATEGORICAL if data[col].nunique()/len(data) < 0.5 else DataType.TEXT
            else:
                profile.feature_types[col] = DataType.CATEGORICAL

    def _calculate_statistics(self, data: dd.DataFrame, profile: DataProfile):
        """Calculate statistics for the dataset and update the profile.

        Args:
            data (dd.DataFrame): The input Dask DataFrame.
            profile (DataProfile): The profile object to update.
        """
        # Missing values
        profile.missing_values = data.isnull().sum().compute().to_dict()
        
        # Unique values (compute on sample)
        sample = data.sample(frac=0.1).compute() if len(data) > 1000 else data.compute()
        profile.unique_values = {col: sample[col].nunique() for col in data.columns}
        
        # Numerical stats (compute on sample)
        num_cols = [col for col, dtype in profile.feature_types.items() if dtype == DataType.NUMERICAL]
        if num_cols:
            profile.stats = {'numerical': sample[num_cols].describe().to_dict()}
        
        # Categorical stats (compute on sample)
        cat_cols = [col for col, dtype in profile.feature_types.items() if dtype == DataType.CATEGORICAL]
        if cat_cols:
            profile.stats['categorical'] = {col: sample[col].value_counts().to_dict() for col in cat_cols}

    def _detect_problem_type(self, data: pd.DataFrame, profile: DataProfile, target_col: str):
        """Detect the problem type based on the target column.

        Args:
            data (pd.DataFrame): A sample of the dataset as a pandas DataFrame.
            profile (DataProfile): The profile object to update.
            target_col (str): The name of the target column.
        """
        target_type = profile.feature_types[target_col]
        if target_type in (DataType.NUMERICAL, DataType.BOOLEAN):
            profile.problem_type = (ProblemType.CLASSIFICATION if data[target_col].nunique() == 2
                                   else ProblemType.REGRESSION)
        else:
            profile.problem_type = ProblemType.CLASSIFICATION
        profile.target_column = target_col

    def _auto_detect_problem_type(self, profile: DataProfile, data: pd.DataFrame) -> str:
        """Auto-detect the problem type when no target column is specified.

        Args:
            profile (DataProfile): The profile object containing feature types.
            data (pd.DataFrame): A sample of the dataset as a pandas DataFrame.

        Returns:
            str: The inferred problem type.
        """
        num_cols = [col for col, dtype in profile.feature_types.items() if dtype == DataType.NUMERICAL]
        return (ProblemType.CLUSTERING if not num_cols or len(num_cols)/len(data.columns) > 0.7
                else ProblemType.CLASSIFICATION)