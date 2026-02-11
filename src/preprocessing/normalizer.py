"""
Text normalization module
Performs stemming, lemmatization, and stopword removal
"""
import nltk
from typing import List
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextNormalizer:
    """Normalize text using stemming, lemmatization, or stopword removal"""
    
    def __init__(self, 
                 use_stemming: bool = False,
                 use_lemmatization: bool = True,
                 remove_stopwords: bool = False):
        """
        Initialize normalizer
        
        Args:
            use_stemming: Apply stemming
            use_lemmatization: Apply lemmatization
            remove_stopwords: Remove stopwords
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        if use_stemming:
            self.stemmer = PorterStemmer()
        
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
    
    def normalize(self, tokens: List[str]) -> List[str]:
        """
        Normalize list of tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Normalized tokens
        """
        normalized = tokens.copy()
        
        # Remove stopwords
        if self.remove_stopwords:
            normalized = [token for token in normalized 
                         if token.lower() not in self.stop_words]
        
        # Apply stemming
        if self.use_stemming:
            normalized = [self.stemmer.stem(token) for token in normalized]
        
        # Apply lemmatization
        if self.use_lemmatization:
            normalized = [self.lemmatizer.lemmatize(token) for token in normalized]
        
        return normalized
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text (tokenizes, normalizes, rejoins)
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        from nltk.tokenize import word_tokenize
        
        tokens = word_tokenize(text)
        normalized_tokens = self.normalize(tokens)
        return ' '.join(normalized_tokens)


# Example usage
if __name__ == "__main__":
    # Test lemmatization
    lemma_normalizer = TextNormalizer(
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=False
    )
    
    tokens = ["running", "runs", "ran", "easily", "fairly"]
    print(f"Original: {tokens}")
    print(f"Lemmatized: {lemma_normalizer.normalize(tokens)}")
    
    # Test stemming
    stem_normalizer = TextNormalizer(
        use_stemming=True,
        use_lemmatization=False,
        remove_stopwords=False
    )
    
    print(f"Stemmed: {stem_normalizer.normalize(tokens)}")
    
    # Test stopword removal
    stopword_normalizer = TextNormalizer(
        use_stemming=False,
        use_lemmatization=False,
        remove_stopwords=True
    )
    
    text_tokens = ["this", "is", "a", "great", "product"]
    print(f"With stopwords: {text_tokens}")
    print(f"Without stopwords: {stopword_normalizer.normalize(text_tokens)}")