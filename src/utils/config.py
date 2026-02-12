"""
Configuration settings for the Sentiment Analysis System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
MODEL_CONFIGS = {
    "naive_bayes": {
        "alpha": 0.1
    },
    "logistic_regression": {
        "C": 2.0,
        "max_iter": 1000,
        "solver": "lbfgs"
    },
    "lstm": {
        "embedding_dim": 128,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1
    },
    "ensemble": {
        "method": "soft",  # Soft voting (probabilities) or hard voting (labels)
        "weights": None    # Optional weights for each model
    },
    "bert": {
        "model_name": "distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 32,  # Reduced for RoBERTa (larger model)
        "epochs": 4,       # Increased for better convergence
        "learning_rate": 2e-5,
        "warmup_steps": 20
    }
}

# TF-IDF settings
TFIDF_CONFIG = {
    "max_features": 20000,
    "ngram_range": (1, 3),
    "min_df": 5,
    "max_df": 0.8
}

# Database settings
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/sentiment"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# API settings
API_KEY_SECRET = os.getenv("API_KEY_SECRET", "your-secret-key-change-this")
API_RATE_LIMIT = 100  # requests per minute

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Sentiment labels
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(SENTIMENT_LABELS)}