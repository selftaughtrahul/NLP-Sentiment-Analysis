"""
Script to download and prepare sentiment analysis dataset
Uses the Twitter Sentiment140 dataset or creates sample data
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add apps directory to sys.path
apps_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(apps_dir))

from sklearn.model_selection import train_test_split
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LABEL_TO_ID
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


from datasets import load_dataset

def download_real_dataset():
    """
    Download a real sentiment analysis dataset
    Uses TweetEval (sentiment)
    """
    logger.info("Downloading TweetEval (sentiment) dataset...")
    
    try:
        # Load dataset
        dataset = load_dataset("tweet_eval", "sentiment")
        
        # Convert to pandas
        # TweetEval has 'train', 'test', 'validation' splits
        # We'll combine them and let our split function handle it to ensure consistency with our pipeline
        train_df = pd.DataFrame(dataset['train'])
        val_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Combine
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        logger.info(f"Successfully loaded TweetEval dataset with {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None


def preprocess_dataset(df):
    """Preprocess the dataset"""
    logger.info("Preprocessing dataset...")
    
    pipeline = PreprocessingPipeline()
    
    # Preprocess texts
    df['processed_text'] = df['text'].apply(
        lambda x: pipeline.preprocess(x, return_tokens=False)
    )
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
    
    logger.info(f"Dataset after preprocessing: {len(df)} samples")
    
    return df


def split_dataset(df, test_size=0.2, val_size=0.1):
    """Split dataset into train, validation, and test sets"""
    logger.info("Splitting dataset...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42,
        stratify=df['label']
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=42,
        stratify=train_val_df['label']
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def save_datasets(train_df, val_df, test_df):
    """Save datasets to CSV files"""
    logger.info("Saving datasets...")
    
    # Ensure directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save with only necessary columns
    columns = ['processed_text', 'label']
    
    # Rename for consistency
    train_df = train_df.rename(columns={'processed_text': 'text'})
    val_df = val_df.rename(columns={'processed_text': 'text'})
    test_df = test_df.rename(columns={'processed_text': 'text'})
    
    train_df[['text', 'label']].to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df[['text', 'label']].to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df[['text', 'label']].to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    logger.info(f"Datasets saved to {PROCESSED_DATA_DIR}")


def main():
    """Main function"""
    logger.info("Starting data preparation...")
    
    # Try to download real dataset, fallback to sample
    df = download_real_dataset()
    
    if df is None:
        logger.warning("Using sample dataset. For production, implement real dataset download.")
        return
    
    # Preprocess
    df = preprocess_dataset(df)
    
    # Split
    train_df, val_df, test_df = split_dataset(df)
    
    # Save
    save_datasets(train_df, val_df, test_df)
    
    logger.info("✅ Data preparation complete!")
    logger.info(f"\nDataset statistics:")
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    logger.info(f"\nFiles saved to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
