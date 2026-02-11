"""
Complete preprocessing pipeline
Combines cleaning, tokenization, and normalization
"""
from typing import List, Dict
from .text_cleaner import TextCleaner
from .tokenizer import Tokenizer
from .normalizer import TextNormalizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PreprocessingPipeline:
    """End-to-end text preprocessing pipeline"""
    
    def __init__(self,
                 clean_params: Dict = None,
                 tokenize_method: str = "word",
                 normalize_params: Dict = None):
        """
        Initialize pipeline
        
        Args:
            clean_params: Parameters for TextCleaner
            tokenize_method: Tokenization method
            normalize_params: Parameters for TextNormalizer
        """
        self.cleaner = TextCleaner()
        self.tokenizer = Tokenizer(method=tokenize_method)
        
        normalize_params = normalize_params or {}
        self.normalizer = TextNormalizer(**normalize_params)
        
        self.clean_params = clean_params or {}
    
    def preprocess(self, text: str, return_tokens: bool = False):
        """
        Preprocess single text
        
        Args:
            text: Input text
            return_tokens: Return tokens instead of text
            
        Returns:
            Preprocessed text or tokens
        """
        # Step 1: Clean
        cleaned = self.cleaner.clean(text, **self.clean_params)
        
        # Step 2: Tokenize
        tokens = self.tokenizer.tokenize(cleaned)
        
        # Step 3: Normalize
        normalized_tokens = self.normalizer.normalize(tokens)
        
        if return_tokens:
            return normalized_tokens
        else:
            return ' '.join(normalized_tokens)
    
    def preprocess_batch(self, texts: List[str], return_tokens: bool = False):
        """
        Preprocess multiple texts
        
        Args:
            texts: List of texts
            return_tokens: Return tokens instead of text
            
        Returns:
            List of preprocessed texts or tokens
        """
        return [self.preprocess(text, return_tokens) for text in texts]


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = PreprocessingPipeline(
        clean_params={
            'remove_urls': True,
            'expand_contractions': True,
            'lowercase': True
        },
        tokenize_method='word',
        normalize_params={
            'use_lemmatization': True,
            'remove_stopwords': False
        }
    )
    
    # Test texts
    test_texts = [
        "I LOVE this product!!! https://example.com",
        "This is terrible. I can't believe it!",
        "It's okay, nothing special really."
    ]
    
    print("Preprocessing Pipeline Examples:")
    print("=" * 60)
    
    for text in test_texts:
        processed = pipeline.preprocess(text)
        tokens = pipeline.preprocess(text, return_tokens=True)
        
        print(f"Original:   {text}")
        print(f"Processed:  {processed}")
        print(f"Tokens:     {tokens}")
        print("-" * 60)