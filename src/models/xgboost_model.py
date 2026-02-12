from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils.config import MODEL_CONFIGS, MODELS_DIR
from src.utils.logger import setup_logger
from pathlib import Path
import pickle

logger = setup_logger(__name__)

class XGBoostSentimentClassifier:
    """XGBoost for sentiment classification"""
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost
        """
        config = MODEL_CONFIGS['xgboost'].copy()
        config.update(kwargs)
        
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            **config
        )
        self.is_trained = False
        logger.info(f"Initialized XGBoost with params: {config}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples...")
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
        
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        logger.info("\n" + report_str)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'report': report_dict
        }
    
    def save(self, filepath: Path = None):
        """Save model"""
        if filepath is None:
            filepath = MODELS_DIR / "xgboost_model.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None):
        """Load model"""
        if filepath is None:
            filepath = MODELS_DIR / "xgboost_model.pkl"
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        classifier = cls()
        classifier.model = model
        classifier.is_trained = True
        
        logger.info(f"Loaded model from {filepath}")
        return classifier
