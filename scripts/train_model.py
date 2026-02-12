"""
Script to train sentiment analysis models
"""
import argparse
import pandas as pd
import sys
from pathlib import Path

# Add apps directory to sys.path
apps_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(apps_dir))

from sklearn.model_selection import train_test_split

from src.preprocessing.pipeline import PreprocessingPipeline
from src.features.tfidf_vectorizer import TfidfFeatureExtractor
from src.models.naive_bayes_model import NaiveBayesClassifier
from src.models.logistic_regression_model import LogisticRegressionClassifier
from src.models.xgboost_model import XGBoostSentimentClassifier
from src.models.ensemble_model import EnsembleSentimentClassifier
from src.models.bert_model import BERTTrainer
from src.utils.config import PROCESSED_DATA_DIR
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load preprocessed data"""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    return train_df, val_df, test_df


def save_evaluation_report(model_name, results, y_true):
    """Save evaluation report and confusion matrix"""
    # Save metrics
    report_path = REPORTS_DIR / f"{model_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(results['report'], f, indent=4)
    logger.info(f"Saved report to {report_path}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = REPORTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {cm_path}")


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
    nb_results = nb.evaluate(X_test, y_test)
    save_evaluation_report("naive_bayes", nb_results, y_test)
    nb.save()
    
    # Train Logistic Regression
    logger.info("\n=== Logistic Regression ===")
    lr = LogisticRegressionClassifier()
    lr.train(X_train, y_train)
    lr_results = lr.evaluate(X_test, y_test)
    save_evaluation_report("logistic_regression", lr_results, y_test)
    lr.save()
    
    # Track trained models for ensemble
    trained_models = [('naive_bayes', nb), ('logistic_regression', lr)]
    
    # Train XGBoost
    logger.info("\n=== XGBoost ===")
    xgb = XGBoostSentimentClassifier()
    xgb.train(X_train, y_train)
    xgb_results = xgb.evaluate(X_test, y_test)
    save_evaluation_report("xgboost", xgb_results, y_test)
    xgb.save()
    trained_models.append(('xgboost', xgb))
    
    # Train Ensemble
    logger.info("\n=== Ensemble (Voting) ===")
    ensemble = EnsembleSentimentClassifier(models=trained_models)
    ensemble.train(X_train, y_train)
    ensemble_results = ensemble.evaluate(X_test, y_test)
    save_evaluation_report("ensemble", ensemble_results, y_test)
    ensemble.save()


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
    bert_results = trainer.evaluate(
        test_df['text'].tolist(),
        test_df['label'].tolist()
    )
    logger.info(f"Test Accuracy: {bert_results['accuracy']:.4f}")
    
    save_evaluation_report("bert", bert_results, test_df['label'].tolist())
    
    # Save
    trainer.save()


def main():
    print("ehllo")
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
    
    logger.info("\n[OK] Training complete!")


if __name__ == "__main__":
    main()