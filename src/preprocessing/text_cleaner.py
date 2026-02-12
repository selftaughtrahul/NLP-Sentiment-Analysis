"""
Text cleaning module for sentiment analysis
Removes URLs, HTML tags, special characters, and normalizes text
"""
import re
import html
from typing import List
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

class TextCleaner:
    """Clean and normalize text data"""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        
        # Contractions mapping
        self.contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "could've": "could have",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where's": "where is",
            "who'd": "who would",
            "who'll": "who will",
            "who're": "who are",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
    
    def clean(self, text: str, 
              remove_urls: bool = True,
              remove_emails: bool = True,
              remove_html: bool = True,
              expand_contractions: bool = True,
              remove_mentions: bool = False,
              remove_hashtags: bool = False,
              remove_numbers: bool = False,
              lowercase: bool = True) -> str:
        """
        Clean text with various options
        
        Args:
            text: Input text
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_html: Remove HTML tags
            expand_contractions: Expand contractions (don't -> do not)
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_numbers: Remove numbers
            lowercase: Convert to lowercase
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        if remove_html:
            text = self.html_pattern.sub('', text)
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove emails
        if remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Remove mentions
        if remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags
        if remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Expand contractions
        if expand_contractions:
            text = self._expand_contractions(text)
        
        # Remove numbers
        if remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Remove special characters (keep letters, numbers, spaces, basic punctuation, @, #)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?@#]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Remove punctuation for matching
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.contractions:
                expanded_words.append(self.contractions[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def clean_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Clean multiple texts
        
        Args:
            texts: List of texts
            **kwargs: Arguments to pass to clean()
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text, **kwargs) for text in texts]


# Example usage
if __name__ == "__main__":
    cleaner = TextCleaner()
    
    # Test cases
    test_texts = [
        "I LOVE this product!!! 😍 https://example.com",
        "This is <b>terrible</b> service. I can't believe it!",
        "Email me at test@example.com for more info",
        "@user Check out #awesome product! It's amazing!!!",
        "I'm loving this! Won't you try it? It's great!!!"
    ]
    
    print("Text Cleaning Examples:")
    print("=" * 60)
    
    for text in test_texts:
        cleaned = cleaner.clean(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 60)