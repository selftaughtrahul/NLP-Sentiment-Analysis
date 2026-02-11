"""
Logistic Regression Sentiment Classifier
"""
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.utils.config import MODEL_CONFIGS, MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LogisticRegressionClassifier:
    """Logistic Regression for sentiment classification"""
    
    def __init__(self, C: float = None, max_iter: int = None):
        """
        Initialize Logistic Regression
        
        Args:
            C: Inverse regularization strength
            max_iter: Maximum iterations
        """
        config = MODEL_CONFIGS['logistic_regression']
        C = C or config['C']
        max_iter = max_iter or config['max_iter']
        solver = config['solver']
        
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            multi_class='multinomial',
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        logger.info(f"Initialized Logistic Regression with C={C}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        logger.info(f"Training Logistic Regression on {X_train.shape[0]} samples...")
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
        
        logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    def save(self, filepath: Path = None):
        """Save model"""
        if filepath is None:
            filepath = MODELS_DIR / "logistic_regression_model.pkl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None):
        """Load model"""
        if filepath is None:
            filepath = MODELS_DIR / "logistic_regression_model.pkl"
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        classifier = cls()
        classifier.model = model
        classifier.is_trained = True
        
        logger.info(f"Loaded model from {filepath}")
        return classifier