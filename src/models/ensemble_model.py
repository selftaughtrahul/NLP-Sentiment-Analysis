from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils.config import MODEL_CONFIGS, MODELS_DIR
from src.utils.logger import setup_logger
from pathlib import Path
import pickle
import numpy as np

logger = setup_logger(__name__)

class EnsembleSentimentClassifier:
    """Voting Ensemble for sentiment classification"""
    
    def __init__(self, models: list = None, voting: str = 'soft', weights: list = None):
        """
        Initialize Ensemble
        
        Args:
            models: List of (name, model) tuples
            voting: 'hard' or 'soft'
            weights: List of weights
        """
        config = MODEL_CONFIGS['ensemble']
        voting = voting or config.get('method', 'soft')
        weights = weights or config.get('weights')
        
        self.voting = voting
        self.weights = weights
        self.estimators = models if models else []
        self.is_trained = False
        
        if models:
            # Extract sklearn estimators from our wrapper classes if needed
            estimators = []
            for name, model_wrapper in models:
                if hasattr(model_wrapper, 'model'):
                    estimators.append((name, model_wrapper.model))
                else:
                    estimators.append((name, model_wrapper))
            
            self.model = VotingClassifier(
                estimators=estimators,
                voting=voting,
                weights=weights,
                n_jobs=-1
            )
            logger.info(f"Initialized Ensemble with {len(models)} models, voting={voting}")
        else:
            logger.warning("Initialized empty Ensemble. Add models before training.")
            self.model = None

    def train(self, X_train, y_train):
        """Train the ensemble"""
        if self.model is None:
            raise ValueError("No models added to ensemble")
            
        logger.info(f"Training Ensemble on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Training complete")
    
    def predict(self, X):
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)
        
        logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        logger.info("\n" + report_str)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'report': report_dict
        }
    
    def save(self, filepath: Path = None):
        """Save ensemble"""
        if filepath is None:
            filepath = MODELS_DIR / "ensemble_model.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Saved ensemble to {filepath}")

    @classmethod
    def load(cls, filepath: Path = None):
        """Load ensemble"""
        # Loading logic is complex for ensemble as it needs sub-estimators.
        # For simplicity, we just load the pickled VotingClassifier
        if filepath is None:
            filepath = MODELS_DIR / "ensemble_model.pkl"
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        classifier = cls()
        classifier.model = model
        classifier.is_trained = True
        
        logger.info(f"Loaded ensemble from {filepath}")
        return classifier
