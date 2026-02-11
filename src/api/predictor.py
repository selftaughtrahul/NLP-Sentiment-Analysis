"""
Sentiment prediction service
Handles model loading and inference
"""
import time
from typing import Dict, List
import numpy as np
from pathlib import Path

from src.preprocessing.pipeline import PreprocessingPipeline
from src.features.tfidf_vectorizer import TfidfFeatureExtractor
from src.models.naive_bayes_model import NaiveBayesClassifier
from src.models.logistic_regression_model import LogisticRegressionClassifier
from src.models.bert_model import BERTTrainer
from src.utils.config import MODELS_DIR, ID_TO_LABEL
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SentimentPredictor:
    """Unified predictor for all models"""
    
    def __init__(self):
        """Initialize predictor and load models"""
        self.models = {}
        self.preprocessing_pipeline = None
        self.tfidf_vectorizer = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        logger.info("Loading models...")
        
        try:
            # Load preprocessing pipeline
            self.preprocessing_pipeline = PreprocessingPipeline(
                clean_params={'lowercase': True, 'expand_contractions': True},
                tokenize_method='word',
                normalize_params={'use_lemmatization': True}
            )
            
            # Load TF-IDF vectorizer (for traditional models)
            tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                self.tfidf_vectorizer = TfidfFeatureExtractor.load(tfidf_path)
                logger.info("[OK] Loaded TF-IDF vectorizer")
            
            # Load Naive Bayes
            nb_path = MODELS_DIR / "naive_bayes_model.pkl"
            if nb_path.exists():
                self.models['naive_bayes'] = NaiveBayesClassifier.load(nb_path)
                logger.info("[OK] Loaded Naive Bayes model")
            
            # Load Logistic Regression
            lr_path = MODELS_DIR / "logistic_regression_model.pkl"
            if lr_path.exists():
                self.models['logistic_regression'] = LogisticRegressionClassifier.load(lr_path)
                logger.info("[OK] Loaded Logistic Regression model")
            
            # Load BERT
            bert_path = MODELS_DIR / "bert_model.pth"
            if bert_path.exists():
                self.models['bert'] = BERTTrainer.load(bert_path)
                logger.info("[OK] Loaded BERT model")
            
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(self, text: str, model_name: str = "bert") -> Dict:
        """
        Predict sentiment for single text
        
        Args:
            text: Input text
            model_name: Model to use
            
        Returns:
            Prediction dictionary
        """
        start_time = time.time()
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. "
                           f"Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Preprocess text
        if model_name in ['naive_bayes', 'logistic_regression']:
            # Traditional models use TF-IDF
            processed_text = self.preprocessing_pipeline.preprocess(text)
            features = self.tfidf_vectorizer.transform([processed_text])
            
            # Predict
            probabilities = model.predict_proba(features)[0]
            predicted_label = np.argmax(probabilities)
            
        else:  # BERT
            # BERT handles its own preprocessing
            probabilities = model.predict_proba([text])[0]
            predicted_label = np.argmax(probabilities)
        
        # Convert to sentiment label
        sentiment = ID_TO_LABEL[predicted_label]
        confidence = float(probabilities[predicted_label])
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'positive': float(probabilities[2])
            },
            'model_used': model_name,
            'processing_time_ms': processing_time
        }
    
    def predict_batch(self, texts: List[str], model_name: str = "bert") -> List[Dict]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts
            model_name: Model to use
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text, model_name) for text in texts]
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models"""
        return list(self.models.keys())


# Global predictor instance
_predictor = None

def get_predictor() -> SentimentPredictor:
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor