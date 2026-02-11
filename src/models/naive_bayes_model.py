"""
Naive Bayes Sentiment Classifier
Baseline model using Multinomial Naive Bayes
"""
import pickle
from pathlib import Path
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.utils.config import MODEL_CONFIGS, MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class NaiveBayesClassifier:
    """Naive Bayes model for sentiment classification"""
    
    def __init__(self, alpha: float = None):
        """
        Initialize Naive Bayes classifier
        
        Args:
            alpha: Smoothing parameter
        """
        alpha = alpha or MODEL_CONFIGS['naive_bayes']['alpha']
        self.model = MultinomialNB(alpha=alpha)
        self.is_trained = False
        logger.info(f"Initialized Naive Bayes with alpha={alpha}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features (TF-IDF vectors)
            y_train: Training labels
        """
        logger.info(f"Training Naive Bayes on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Training complete")
    
    def predict(self, X):
        """
        Predict sentiment labels
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)
        
        logger.info(f"Naive Bayes Accuracy: {accuracy:.4f}")
        logger.info("\n" + report_str)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'report': report_dict
        }
    
    def save(self, filepath: Path = None):
        """Save model to disk"""
        if filepath is None:
            filepath = MODELS_DIR / "naive_bayes_model.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None):
        """Load model from disk"""
        if filepath is None:
            filepath = MODELS_DIR / "naive_bayes_model.pkl"
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        classifier = cls()
        classifier.model = model
        classifier.is_trained = True
        
        logger.info(f"Loaded model from {filepath}")
        return classifier