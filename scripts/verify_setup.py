"""
Verify project setup
"""
import sys
from pathlib import Path

def check_directories():
    """Check if all required directories exist"""
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/models",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/api",
        "src/utils",
        "scripts",
        "tests",
        "notebooks",
        "dashboard",
        "logs"
    ]
    
    print("Checking directories...")
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_imports():
    """Check if required packages are installed"""
    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "nltk",
        "spacy",
        "torch",
        "transformers",
        "fastapi",
        "streamlit"
    ]
    
    print("\nChecking package installations...")
    all_installed = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_nltk_data():
    """Check if NLTK data is downloaded"""
    import nltk
    
    print("\nChecking NLTK data...")
    resources = ['punkt', 'stopwords', 'wordnet']
    all_downloaded = True
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                          else f'corpora/{resource}')
            print(f"✓ {resource}")
        except LookupError:
            print(f"✗ {resource} - NOT DOWNLOADED")
            all_downloaded = False
    
    return all_downloaded

if __name__ == "__main__":
    print("=" * 50)
    print("VERIFYING PROJECT SETUP")
    print("=" * 50)
    
    dirs_ok = check_directories()
    packages_ok = check_imports()
    nltk_ok = check_nltk_data()
    
    print("\n" + "=" * 50)
    if dirs_ok and packages_ok and nltk_ok:
        print("✅ ALL CHECKS PASSED!")
        print("Your environment is ready for development.")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please fix the issues above before proceeding.")
        sys.exit(1)