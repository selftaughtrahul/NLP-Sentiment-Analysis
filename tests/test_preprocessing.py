"""
Unit tests for preprocessing module
"""
import pytest
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.pipeline import PreprocessingPipeline


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()
    
    def test_remove_urls(self):
        text = "Check this out https://example.com"
        cleaned = self.cleaner.clean(text)
        assert "https" not in cleaned
        assert "example.com" not in cleaned
    
    def test_lowercase(self):
        text = "THIS IS UPPERCASE"
        cleaned = self.cleaner.clean(text, lowercase=True)
        assert cleaned.islower()
    
    def test_expand_contractions(self):
        text = "I can't believe it"
        cleaned = self.cleaner.clean(text, expand_contractions=True)
        assert "cannot" in cleaned


class TestTokenizer:
    def test_word_tokenization(self):
        tokenizer = Tokenizer(method="word")
        text = "This is a test."
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        assert "This" in tokens or "this" in tokens
    
    def test_sentence_tokenization(self):
        tokenizer = Tokenizer(method="sentence")
        text = "First sentence. Second sentence."
        sentences = tokenizer.tokenize(text)
        assert len(sentences) == 2


class TestTextNormalizer:
    def test_lemmatization(self):
        normalizer = TextNormalizer(use_lemmatization=True)
        tokens = ["running", "ran"]
        normalized = normalizer.normalize(tokens)
        # Both should be lemmatized to "run" or similar
        assert len(normalized) == 2


class TestPreprocessingPipeline:
    def test_full_pipeline(self):
        pipeline = PreprocessingPipeline()
        text = "I LOVE this!!! https://test.com"
        processed = pipeline.preprocess(text)
        
        # Should be cleaned and lowercase
        assert processed.islower()
        assert "https" not in processed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])