"""
Unit tests for feature extraction module
"""
import pytest
import numpy as np
from src.features.tfidf_vectorizer import TfidfFeatureExtractor
from src.features.embeddings import BERTEmbeddings


class TestTfidfFeatureExtractor:
    """Test TF-IDF feature extraction"""
    
    def setup_method(self):
        """Setup test data"""
        self.texts = [
            "this is a great product",
            "terrible service very bad",
            "average quality nothing special",
            "amazing quality highly recommend",
            "worst purchase ever"
        ]
    
    def test_initialization(self):
        """Test vectorizer initialization"""
        extractor = TfidfFeatureExtractor(
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        assert extractor.vectorizer is not None
        assert not extractor.is_fitted
    
    def test_fit_transform(self):
        """Test fit_transform method"""
        extractor = TfidfFeatureExtractor(
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        vectors = extractor.fit_transform(self.texts)
        
        assert vectors.shape[0] == len(self.texts)
        assert vectors.shape[1] > 0
        assert extractor.is_fitted
    
    def test_fit_then_transform(self):
        """Test separate fit and transform"""
        extractor = TfidfFeatureExtractor(
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        # Fit on training data
        extractor.fit(self.texts)
        assert extractor.is_fitted
        
        # Transform new text
        new_text = ["great product"]
        vector = extractor.transform(new_text)
        
        assert vector.shape[0] == 1
        assert vector.shape[1] > 0
    
    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises error"""
        extractor = TfidfFeatureExtractor(
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        with pytest.raises(ValueError, match="not fitted"):
            extractor.transform(["test"])
    
    def test_get_feature_names(self):
        """Test getting feature names"""
        extractor = TfidfFeatureExtractor(
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        extractor.fit(self.texts)
        
        features = extractor.get_feature_names()
        assert len(features) > 0
        assert isinstance(features[0], str)
    
    def test_get_feature_names_before_fit_raises_error(self):
        """Test that get_feature_names before fit raises error"""
        extractor = TfidfFeatureExtractor(
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        
        with pytest.raises(ValueError, match="not fitted"):
            extractor.get_feature_names()
    
    def test_save_and_load(self, tmp_path):
        """Test save and load functionality"""
        extractor = TfidfFeatureExtractor(
            max_features=100,
            min_df=1,
            max_df=1.0
        )
        extractor.fit_transform(self.texts)
        
        # Save
        save_path = tmp_path / "test_tfidf.pkl"
        extractor.save(save_path)
        assert save_path.exists()
        
        # Load
        loaded_extractor = TfidfFeatureExtractor.load(save_path)
        assert loaded_extractor.is_fitted
        
        # Test loaded vectorizer works
        new_text = ["great product"]
        vector = loaded_extractor.transform(new_text)
        assert vector.shape[0] == 1


class TestBERTEmbeddings:
    """Test BERT embeddings extraction"""
    
    @pytest.mark.slow
    def test_initialization(self):
        """Test BERT model initialization"""
        bert = BERTEmbeddings()
        assert bert.tokenizer is not None
        assert bert.model is not None
        assert bert.device is not None
    
    @pytest.mark.slow
    def test_single_embedding(self):
        """Test single text embedding"""
        bert = BERTEmbeddings()
        text = "This is a test"
        
        embedding = bert.get_embedding(text)
        
        # BERT base produces 768-dim embeddings
        assert embedding.shape == (768,)
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64
    
    @pytest.mark.slow
    def test_batch_embeddings(self):
        """Test batch embedding generation"""
        bert = BERTEmbeddings()
        texts = ["First text", "Second text", "Third text"]
        
        embeddings = bert.get_embeddings_batch(texts)
        
        assert embeddings.shape == (3, 768)
        assert isinstance(embeddings, np.ndarray)
    
    @pytest.mark.slow
    def test_embedding_consistency(self):
        """Test that same text produces same embedding"""
        bert = BERTEmbeddings()
        text = "Consistent test text"
        
        embedding1 = bert.get_embedding(text)
        embedding2 = bert.get_embedding(text)
        
        # Should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
    
    @pytest.mark.slow
    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings"""
        bert = BERTEmbeddings()
        text1 = "This is great"
        text2 = "This is terrible"
        
        embedding1 = bert.get_embedding(text1)
        embedding2 = bert.get_embedding(text2)
        
        # Embeddings should be different
        assert not np.array_equal(embedding1, embedding2)
    
    @pytest.mark.slow
    def test_max_length_parameter(self):
        """Test max_length parameter"""
        bert = BERTEmbeddings()
        long_text = " ".join(["word"] * 1000)  # Very long text
        
        # Should not raise error even with very long text
        embedding = bert.get_embedding(long_text, max_length=128)
        assert embedding.shape == (768,)
    
    @pytest.mark.slow
    def test_empty_text_handling(self):
        """Test handling of empty text"""
        bert = BERTEmbeddings()
        
        # Empty string should still produce embedding
        embedding = bert.get_embedding("")
        assert embedding.shape == (768,)


# Configuration for pytest
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
