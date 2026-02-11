"""
Script to download and prepare sentiment analysis dataset
Uses the Twitter Sentiment140 dataset or creates sample data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LABEL_TO_ID
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_sample_dataset():
    """
    Create a sample dataset for testing
    In production, replace this with actual dataset loading
    """
    logger.info("Creating sample dataset...")
    
    # Sample data (replace with actual dataset)
    samples = [
        # Positive samples
        ("I absolutely love this product! It's amazing and works perfectly.", 2),
        ("This is the best purchase I've ever made. Highly recommend!", 2),
        ("Excellent quality and fast shipping. Very satisfied!", 2),
        ("Great value for money. Exceeded my expectations!", 2),
        ("Outstanding service and product. Will buy again!", 2),
        ("Fantastic! Exactly what I needed. Five stars!", 2),
        ("Love it! Works great and looks beautiful.", 2),
        ("Perfect! No complaints at all. Very happy!", 2),
        ("Wonderful experience. The product is excellent.", 2),
        ("Amazing quality! Totally worth the price.", 2),
        
        # Negative samples
        ("Terrible product. Complete waste of money.", 0),
        ("Very disappointed. Does not work as advertised.", 0),
        ("Worst purchase ever. Broke after one day.", 0),
        ("Poor quality. Would not recommend to anyone.", 0),
        ("Horrible experience. Customer service was awful.", 0),
        ("Don't buy this! It's a scam.", 0),
        ("Completely useless. Returning immediately.", 0),
        ("Awful product. Save your money.", 0),
        ("Terrible quality and overpriced.", 0),
        ("Worst experience ever. Very unhappy.", 0),
        
        # Neutral samples
        ("It's okay. Nothing special but works fine.", 1),
        ("Average product. Does what it says.", 1),
        ("Not bad, not great. Just okay.", 1),
        ("Decent quality for the price.", 1),
        ("It works. No major issues.", 1),
        ("Acceptable. Meets basic expectations.", 1),
        ("Fair product. Could be better.", 1),
        ("Standard quality. Nothing remarkable.", 1),
        ("It's fine. Does the job.", 1),
        ("Okay product. Average experience.", 1),
    ]
    
    # Expand dataset by creating variations
    expanded_samples = []
    for text, label in samples:
        expanded_samples.append((text, label))
        # Add slight variations
        expanded_samples.append((text.lower(), label))
        expanded_samples.append((text + " Really!", label))
    
    # Create more samples to have a reasonable dataset size
    # In production, you would load a real dataset here
    for _ in range(100):  # Replicate to get ~300 samples
        for text, label in samples:
            expanded_samples.append((text, label))
    
    df = pd.DataFrame(expanded_samples, columns=['text', 'label'])
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Created dataset with {len(df)} samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df


def download_real_dataset():
    """
    Download a real sentiment analysis dataset
    Options:
    1. IMDB Reviews
    2. Twitter Sentiment140
    3. Amazon Reviews
    
    For now, returns None - implement based on your needs
    """
    # TODO: Implement actual dataset download
    # Example using datasets library:
    # from datasets import load_dataset
    # dataset = load_dataset("imdb")
    # df = pd.DataFrame(dataset['train'])
    
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
        df = create_sample_dataset()
    
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
