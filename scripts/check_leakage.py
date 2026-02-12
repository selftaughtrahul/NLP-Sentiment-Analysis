import pandas as pd
import sys
from pathlib import Path

# Add apps directory to sys.path
apps_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(apps_dir))

from src.utils.config import PROCESSED_DATA_DIR

def check_leakage():
    print("Checking for data leakage...")
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("Data files not found.")
        return
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Check intersection
    train_texts = set(train_df['text'].unique())
    test_texts = set(test_df['text'].unique())
    
    intersection = train_texts.intersection(test_texts)
    
    print(f"Unique texts in Train: {len(train_texts)}")
    print(f"Unique texts in Test: {len(test_texts)}")
    print(f"Overlapping unique texts: {len(intersection)}")
    
    leakage_percent = len(intersection) / len(test_texts) * 100 if len(test_texts) > 0 else 0
    print(f"Percentage of test usage that is also in train: {leakage_percent:.2f}%")
    
    if leakage_percent > 0:
        print("\nDATA LEAKAGE DETECTED!")
        print("The model has seen these test examples during training.")

if __name__ == "__main__":
    check_leakage()
