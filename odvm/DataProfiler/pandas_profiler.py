import pandas as pd
import numpy as np
from typing import Union, Optional
from .base import BaseDataProfiler, DataProfile
from ..constants import DataType, ProblemType
from ..exceptions import DataProfilingError, DataValidationError

class PandasDataProfiler(BaseDataProfiler):
    """
    Data profiler implementation utilizing pandas for in-memory analysis of datasets.

    This class generates a comprehensive profile of a pandas DataFrame, including feature type detection,
    missing and unique value analysis, statistical summaries, and problem type inference. It is suitable
    for small to medium-sized datasets that fit in memory.

    Methods:
        profile(data, target_col):
            Generates a DataProfile for the provided pandas DataFrame, optionally using a specified target column.

    Raises:
        DataProfilingError: If profiling fails due to unexpected errors.
        DataValidationError: If validation of the input data or target column fails.
    """
    
    def profile(self, data: pd.DataFrame, target_col: Optional[str] = None) -> DataProfile:
        """Generate a comprehensive data profile for a pandas DataFrame.

        Args:
            data (pd.DataFrame): The input pandas DataFrame to be profiled.
            target_col (Optional[str]): The name of the target column, if applicable.

        Returns:
            DataProfile: The generated data profile containing feature types, statistics, and metadata.

        Raises:
            DataProfilingError: If profiling fails.
        """
        try:
            self._validate_input_data(data)
            profile = DataProfile()
            
            # Detect feature types
            self._detect_feature_types(data, profile)
            
            # Calculate statistics
            self._calculate_statistics(data, profile)
            
            # Detect problem type
            if target_col:
                self._validate_target_column(data, target_col)
                self._detect_problem_type(data, profile, target_col)
            else:
                profile.problem_type = self._auto_detect_problem_type(profile, data)
                
            return profile
        except Exception as e:
            raise DataProfilingError(f"Profiling failed: {str(e)}") from e

    def _validate_target_column(self, data: pd.DataFrame, target_col: str) -> None:
        """Validate the target column exists and has valid data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target_col (str): The name of the target column.

        Raises:
            DataValidationError: If the target column is missing or contains only null values.
        """
        if target_col not in data.columns:
            raise DataValidationError(f"Target column '{target_col}' not found in data")
        if data[target_col].isnull().all():
            raise DataValidationError(f"Target column '{target_col}' contains only null values")

    def _detect_feature_types(self, data: pd.DataFrame, profile: DataProfile):
        """Detect and record feature types for each column.

        Args:
            data (pd.DataFrame): The input DataFrame.
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

    def _calculate_statistics(self, data: pd.DataFrame, profile: DataProfile):
        """Calculate dataset statistics and update the profile.

        Args:
            data (pd.DataFrame): The input DataFrame.
            profile (DataProfile): The profile object to update.
        """
        # Missing values
        profile.missing_values = data.isnull().sum().to_dict()
        
        # Unique values
        profile.unique_values = {col: data[col].nunique() for col in data.columns}
        
        # Numerical stats
        num_cols = [col for col, dtype in profile.feature_types.items() if dtype == DataType.NUMERICAL]
        if num_cols:
            profile.stats = {'numerical': data[num_cols].describe().to_dict()}
        
        # Categorical stats
        cat_cols = [col for col, dtype in profile.feature_types.items() if dtype == DataType.CATEGORICAL]
        if cat_cols:
            profile.stats['categorical'] = {col: data[col].value_counts().to_dict() for col in cat_cols}

    def _detect_problem_type(self, data: pd.DataFrame, profile: DataProfile, target_col: str):
        """Detect problem type based on the target column.

        Args:
            data (pd.DataFrame): The input DataFrame.
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
        """Auto-detect problem type when no target is specified.

        Args:
            profile (DataProfile): The profile object containing feature types.
            data (pd.DataFrame): The input DataFrame.

        Returns:
            str: The inferred problem type.
        """
        num_cols = [col for col, dtype in profile.feature_types.items() if dtype == DataType.NUMERICAL]
        return (ProblemType.CLUSTERING if not num_cols or len(num_cols)/len(data.columns) > 0.7
                else ProblemType.CLASSIFICATION)