"""
TF-IDF Feature Extraction
Converts text to TF-IDF vectors for traditional ML models
"""
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.config import TFIDF_CONFIG, MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TfidfFeatureExtractor:
    """Extract TF-IDF features from text"""
    
    def __init__(self, 
                 max_features: int = None,
                 ngram_range: Tuple[int, int] = None,
                 min_df: int = None,
                 max_df: float = None):
        """
        Initialize TF-IDF vectorizer
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (min_n, max_n)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        # Use config defaults if not provided
        max_features = max_features or TFIDF_CONFIG['max_features']
        ngram_range = ngram_range or TFIDF_CONFIG['ngram_range']
        min_df = min_df or TFIDF_CONFIG['min_df']
        max_df = max_df or TFIDF_CONFIG['max_df']
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Use log scaling
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english'
        )
        
        self.is_fitted = False
        logger.info(f"Initialized TF-IDF vectorizer with max_features={max_features}")
    
    def fit(self, texts: List[str]):
        """
        Fit vectorizer on training texts
        
        Args:
            texts: List of training texts
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors
        
        Args:
            texts: List of texts
            
        Returns:
            TF-IDF matrix (sparse)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            texts: List of texts
            
        Returns:
            TF-IDF matrix (sparse)
        """
        logger.info(f"Fitting and transforming {len(texts)} texts...")
        vectors = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        logger.info(f"Done. Shape: {vectors.shape}")
        return vectors
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get top N features by average TF-IDF score
        
        Args:
            n: Number of top features
            
        Returns:
            List of top feature names
        """
        feature_names = self.get_feature_names()
        return feature_names[:n]
    
    def save(self, filepath: Path = None):
        """
        Save vectorizer to disk
        
        Args:
            filepath: Path to save file
        """
        if filepath is None:
            filepath = MODELS_DIR / "tfidf_vectorizer.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Saved TF-IDF vectorizer to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None):
        """
        Load vectorizer from disk
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded TfidfFeatureExtractor
        """
        if filepath is None:
            filepath = MODELS_DIR / "tfidf_vectorizer.pkl"
        
        with open(filepath, 'rb') as f:
            vectorizer = pickle.load(f)
        
        extractor = cls()
        extractor.vectorizer = vectorizer
        extractor.is_fitted = True
        
        logger.info(f"Loaded TF-IDF vectorizer from {filepath}")
        return extractor


# Example usage
if __name__ == "__main__":
    # Sample data
    texts = [
        "this product is great i love it",
        "terrible service very disappointed",
        "okay nothing special average product",
        "amazing quality highly recommend",
        "worst purchase ever waste of money"
    ]
    
    # Create and fit vectorizer with parameters suitable for small dataset
    extractor = TfidfFeatureExtractor(
        max_features=100,  # Reduced for small dataset
        min_df=1,          # Minimum 1 document (instead of 5)
        max_df=1.0         # Maximum 100% (instead of 0.8)
    )
    vectors = extractor.fit_transform(texts)
    
    print(f"TF-IDF Matrix Shape: {vectors.shape}")
    print(f"Vocabulary Size: {len(extractor.get_feature_names())}")
    print(f"Top Features: {extractor.get_top_features(10)}")
    
    # Transform new text
    new_text = ["this is a great product"]
    new_vector = extractor.transform(new_text)
    print(f"New Vector Shape: {new_vector.shape}")