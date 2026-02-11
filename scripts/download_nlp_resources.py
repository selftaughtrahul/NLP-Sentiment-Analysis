"""
Download required NLP resources
"""
import nltk
import spacy
import subprocess

def download_nltk_data():
    """Download NLTK datasets"""
    print("Downloading NLTK data...")
    
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource)
            print(f"✓ Downloaded {resource}")
        except Exception as e:
            print(f"✗ Failed to download {resource}: {e}")

def download_spacy_model():
    """Download spaCy model"""
    print("\nDownloading spaCy model...")
    
    try:
        subprocess.run([
            "python", "-m", "spacy", "download", "en_core_web_sm"
        ], check=True)
        print("✓ Downloaded en_core_web_sm")
    except Exception as e:
        print(f"✗ Failed to download spaCy model: {e}")

if __name__ == "__main__":
    download_nltk_data()
    download_spacy_model()
    print("\n✅ All NLP resources downloaded successfully!")