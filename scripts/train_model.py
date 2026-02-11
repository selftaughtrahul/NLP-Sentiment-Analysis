"""
Script to train sentiment analysis models
"""
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.preprocessing.pipeline import PreprocessingPipeline
from src.features.tfidf_vectorizer import TfidfFeatureExtractor
from src.models.naive_bayes_model import NaiveBayesClassifier
from src.models.logistic_regression_model import LogisticRegressionClassifier
from src.models.bert_model import BERTTrainer
from src.utils.config import PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data():
    """Load preprocessed data"""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    return train_df, val_df, test_df


def train_traditional_models(train_df, test_df):
    """Train Naive Bayes and Logistic Regression"""
    logger.info("Training traditional ML models...")
    
    # Extract features with TF-IDF
    tfidf = TfidfFeatureExtractor()
    X_train = tfidf.fit_transform(train_df['text'].tolist())
    X_test = tfidf.transform(test_df['text'].tolist())
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # Save TF-IDF vectorizer
    tfidf.save()
    
    # Train Naive Bayes
    logger.info("\n=== Naive Bayes ===")
    nb = NaiveBayesClassifier()
    nb.train(X_train, y_train)
    nb.evaluate(X_test, y_test)
    nb.save()
    
    # Train Logistic Regression
    logger.info("\n=== Logistic Regression ===")
    lr = LogisticRegressionClassifier()
    lr.train(X_train, y_train)
    lr.evaluate(X_test, y_test)
    lr.save()


def train_bert_model(train_df, val_df, test_df):
    """Train BERT model"""
    logger.info("\n=== BERT Model ===")
    
    trainer = BERTTrainer(n_classes=3)
    
    trainer.train(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        val_df['text'].tolist(),
        val_df['label'].tolist()
    )
    
    # Evaluate
    test_acc = trainer.evaluate(
        test_df['text'].tolist(),
        test_df['label'].tolist()
    )
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Save
    trainer.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',choices=['all', 'traditional', 'bert'], help='Which models to train')
    args = parser.parse_args()
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train models
    if args.model in ['all', 'traditional']:
        train_traditional_models(train_df, test_df)
    
    if args.model in ['all', 'bert']:
        train_bert_model(train_df, val_df, test_df)
    
    logger.info("\n✅ Training complete!")


if __name__ == "__main__":
    main()