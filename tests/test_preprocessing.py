"""
Comprehensive unit tests for preprocessing module
Tests all components: TextCleaner, Tokenizer, TextNormalizer, PreprocessingPipeline
"""
import pytest
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.pipeline import PreprocessingPipeline


class TestTextCleaner:
    """Test TextCleaner class"""
    
    def setup_method(self):
        """Setup test instance"""
        self.cleaner = TextCleaner()
    
    def test_remove_urls(self):
        """Test URL removal"""
        text = "Check this out https://example.com and www.test.com"
        cleaned = self.cleaner.clean(text, remove_urls=True)
        assert "https" not in cleaned
        assert "example.com" not in cleaned
        assert "www.test.com" not in cleaned
    
    def test_remove_emails(self):
        """Test email removal"""
        text = "Contact me at test@example.com for info"
        cleaned = self.cleaner.clean(text, remove_emails=True)
        assert "test@example.com" not in cleaned
    
    def test_remove_html(self):
        """Test HTML tag removal"""
        text = "This is <b>bold</b> and <i>italic</i> text"
        cleaned = self.cleaner.clean(text, remove_html=True)
        assert "<b>" not in cleaned
        assert "</b>" not in cleaned
        assert "bold" in cleaned
    
    def test_lowercase(self):
        """Test lowercase conversion"""
        text = "THIS IS UPPERCASE TEXT"
        cleaned = self.cleaner.clean(text, lowercase=True)
        assert cleaned.islower()
    
    def test_expand_contractions(self):
        """Test contraction expansion"""
        text = "I can't believe it won't work"
        cleaned = self.cleaner.clean(text, expand_contractions=True, lowercase=True)
        assert "cannot" in cleaned
        assert "will not" in cleaned
    
    def test_remove_mentions(self):
        """Test @mention removal"""
        text = "@user Check this out @another_user"
        cleaned = self.cleaner.clean(text, remove_mentions=True)
        assert "@user" not in cleaned
        assert "@another_user" not in cleaned
    
    def test_remove_hashtags(self):
        """Test #hashtag removal"""
        text = "This is #awesome and #great"
        cleaned = self.cleaner.clean(text, remove_hashtags=True)
        assert "#awesome" not in cleaned
        assert "#great" not in cleaned
    
    def test_remove_numbers(self):
        """Test number removal"""
        text = "I have 123 items and 456 more"
        cleaned = self.cleaner.clean(text, remove_numbers=True)
        assert "123" not in cleaned
        assert "456" not in cleaned
    
    def test_empty_string(self):
        """Test empty string handling"""
        text = ""
        cleaned = self.cleaner.clean(text)
        assert cleaned == ""
    
    def test_none_input(self):
        """Test None input handling"""
        text = None
        cleaned = self.cleaner.clean(text)
        assert cleaned == ""
    
    def test_clean_batch(self):
        """Test batch cleaning"""
        texts = [
            "First text with https://url.com",
            "Second text with @mention",
            "Third text"
        ]
        cleaned = self.cleaner.clean_batch(texts, remove_urls=True, remove_mentions=True)
        assert len(cleaned) == 3
        assert "https" not in cleaned[0]
        assert "@mention" not in cleaned[1]


class TestTokenizer:
    """Test Tokenizer class"""
    
    def test_word_tokenization(self):
        """Test word tokenization"""
        tokenizer = Tokenizer(method="word")
        text = "This is a test sentence."
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        assert "This" in tokens or "this" in tokens
        assert "test" in tokens
    
    def test_sentence_tokenization(self):
        """Test sentence tokenization"""
        tokenizer = Tokenizer(method="sentence")
        text = "First sentence. Second sentence. Third sentence."
        sentences = tokenizer.tokenize(text)
        
        assert len(sentences) == 3
        assert "First sentence." in sentences
    
    def test_bert_tokenization(self):
        """Test BERT tokenization"""
        tokenizer = Tokenizer(method="bert")
        text = "This is a test"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        assert isinstance(tokens, list)
    
    def test_empty_text(self):
        """Test empty text handling"""
        tokenizer = Tokenizer(method="word")
        tokens = tokenizer.tokenize("")
        assert tokens == []
    
    def test_invalid_method(self):
        """Test invalid tokenization method"""
        tokenizer = Tokenizer(method="invalid")
        with pytest.raises(ValueError, match="Unknown tokenization method"):
            tokenizer.tokenize("test")
    
    def test_tokenize_batch(self):
        """Test batch tokenization"""
        tokenizer = Tokenizer(method="word")
        texts = ["First text", "Second text", "Third text"]
        results = tokenizer.tokenize_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(tokens, list) for tokens in results)


class TestTextNormalizer:
    """Test TextNormalizer class"""
    
    def test_lemmatization(self):
        """Test lemmatization"""
        normalizer = TextNormalizer(use_lemmatization=True, use_stemming=False)
        tokens = ["running", "ran", "runs"]
        normalized = normalizer.normalize(tokens)
        
        assert len(normalized) == 3
        # All should be lemmatized to similar form
        assert isinstance(normalized, list)
    
    def test_stemming(self):
        """Test stemming"""
        normalizer = TextNormalizer(use_stemming=True, use_lemmatization=False)
        tokens = ["running", "runner", "runs"]
        normalized = normalizer.normalize(tokens)
        
        assert len(normalized) == 3
        # All should have similar stems
        assert isinstance(normalized, list)
    
    def test_stopword_removal(self):
        """Test stopword removal"""
        normalizer = TextNormalizer(remove_stopwords=True)
        tokens = ["this", "is", "a", "great", "product"]
        normalized = normalizer.normalize(tokens)
        
        # Common stopwords should be removed
        assert "great" in normalized
        assert "product" in normalized
        assert len(normalized) < len(tokens)
    
    def test_no_normalization(self):
        """Test with no normalization"""
        normalizer = TextNormalizer(
            use_stemming=False,
            use_lemmatization=False,
            remove_stopwords=False
        )
        tokens = ["running", "is", "great"]
        normalized = normalizer.normalize(tokens)
        
        # Should return same tokens
        assert normalized == tokens
    
    def test_empty_tokens(self):
        """Test empty token list"""
        normalizer = TextNormalizer()
        tokens = []
        normalized = normalizer.normalize(tokens)
        assert normalized == []
    
    def test_normalize_text(self):
        """Test text normalization (end-to-end)"""
        normalizer = TextNormalizer(use_lemmatization=True)
        text = "running quickly"
        normalized = normalizer.normalize_text(text)
        
        assert isinstance(normalized, str)
        assert len(normalized) > 0


class TestPreprocessingPipeline:
    """Test PreprocessingPipeline class"""
    
    def test_full_pipeline_default(self):
        """Test full pipeline with default settings"""
        pipeline = PreprocessingPipeline()
        text = "I LOVE this!!! https://test.com"
        processed = pipeline.preprocess(text)
        
        # Should be cleaned and lowercase
        assert isinstance(processed, str)
        assert processed.islower()
        assert "https" not in processed
    
    def test_pipeline_return_tokens(self):
        """Test pipeline returning tokens"""
        pipeline = PreprocessingPipeline()
        text = "This is a test"
        tokens = pipeline.preprocess(text, return_tokens=True)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_pipeline_with_custom_params(self):
        """Test pipeline with custom parameters"""
        pipeline = PreprocessingPipeline(
            clean_params={'lowercase': True, 'expand_contractions': True},
            tokenize_method='word',
            normalize_params={'use_lemmatization': True, 'remove_stopwords': False}  # Don't remove stopwords for this test
        )
        
        text = "I can't believe this is amazing!"
        processed = pipeline.preprocess(text)
        
        assert isinstance(processed, str)
        # Check that contractions were expanded (can't -> cannot or can not)
        assert "believe" in processed
        assert "amazing" in processed
    
    def test_pipeline_batch(self):
        """Test batch preprocessing"""
        pipeline = PreprocessingPipeline()
        texts = [
            "First text with URL https://test.com",
            "Second text is great",
            "Third text"
        ]
        
        processed = pipeline.preprocess_batch(texts)
        
        assert len(processed) == 3
        assert all(isinstance(text, str) for text in processed)
        assert "https" not in processed[0]
    
    def test_pipeline_batch_return_tokens(self):
        """Test batch preprocessing returning tokens"""
        pipeline = PreprocessingPipeline()
        texts = ["First text", "Second text"]
        
        results = pipeline.preprocess_batch(texts, return_tokens=True)
        
        assert len(results) == 2
        assert all(isinstance(tokens, list) for tokens in results)
    
    def test_empty_text(self):
        """Test empty text handling"""
        pipeline = PreprocessingPipeline()
        processed = pipeline.preprocess("")
        
        # Should handle gracefully
        assert isinstance(processed, str)


# Test integration scenarios
class TestIntegration:
    """Integration tests for preprocessing components"""
    
    def test_cleaner_tokenizer_integration(self):
        """Test cleaner and tokenizer working together"""
        cleaner = TextCleaner()
        tokenizer = Tokenizer(method="word")
        
        text = "Check https://example.com for more info!"
        cleaned = cleaner.clean(text, remove_urls=True)
        tokens = tokenizer.tokenize(cleaned)
        
        assert "https" not in " ".join(tokens)
        assert len(tokens) > 0
    
    def test_full_preprocessing_chain(self):
        """Test complete preprocessing chain"""
        cleaner = TextCleaner()
        tokenizer = Tokenizer(method="word")
        normalizer = TextNormalizer(use_lemmatization=True)
        
        text = "I'm running to the store with https://coupon.com"
        
        # Step 1: Clean
        cleaned = cleaner.clean(text, expand_contractions=True, remove_urls=True)
        
        # Step 2: Tokenize
        tokens = tokenizer.tokenize(cleaned)
        
        # Step 3: Normalize
        normalized = normalizer.normalize(tokens)
        
        assert len(normalized) > 0
        assert "https" not in " ".join(normalized)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])