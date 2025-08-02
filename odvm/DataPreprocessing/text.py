from typing import List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseDataPreprocessor
from ..exceptions import DataPreprocessingError

class TextPreprocessor(BaseDataPreprocessor):
    """
    Preprocessor for handling text features.

    This class provides preprocessing for text data using vectorization techniques.
    By default, it uses TF-IDF vectorization via scikit-learn's TfidfVectorizer.
    The vectorization method can be configured during initialization.

    Args:
        vectorization (str): The vectorization strategy to use. Default is 'tfidf'.

    Methods:
        preprocess(data, text_features):
            Preprocesses the specified text features in the input data using the configured vectorizer.

    Raises:
        DataPreprocessingError: If preprocessing fails.
    """
    
    def __init__(self, vectorization: str = 'tfidf'):
        super().__init__()
        self.vectorization = vectorization
    
    def preprocess(self, data, text_features: List[str]):
        """Preprocess text features.

        Applies vectorization to the specified text features using the configured vectorizer.

        Args:
            data: The input DataFrame containing the text features.
            text_features (List[str]): List of text feature names to preprocess.

        Returns:
            The processed text features as a sparse matrix or array, depending on the vectorizer.

        Raises:
            DataPreprocessingError: If preprocessing fails.
        """
        try:
            if not text_features:
                return data
                
            self.preprocessor = Pipeline([
                ('vectorizer', TfidfVectorizer())
            ])
            processed = self.preprocessor.fit_transform(data[text_features[0]])
            
            return processed
        except Exception as e:
            raise DataPreprocessingError(f"Text preprocessing failed: {str(e)}") from e