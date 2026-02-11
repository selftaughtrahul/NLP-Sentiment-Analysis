"""
Tokenization module for text processing
Supports word, sentence, and subword tokenization
"""
import nltk
from typing import List
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BertTokenizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Tokenizer:
    """Tokenize text into words or sentences"""
    
    def __init__(self, method: str = "word"):
        """
        Initialize tokenizer
        
        Args:
            method: Tokenization method ('word', 'sentence', 'bert')
        """
        self.method = method
        
        if method == "bert":
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text based on selected method
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        if self.method == "word":
            return word_tokenize(text)
        elif self.method == "sentence":
            return sent_tokenize(text)
        elif self.method == "bert":
            return self.bert_tokenizer.tokenize(text)
        else:
            raise ValueError(f"Unknown tokenization method: {self.method}")
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of token lists
        """
        return [self.tokenize(text) for text in texts]
    
    def encode_for_bert(self, text: str, max_length: int = 512) -> dict:
        """
        Encode text for BERT model
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        if self.method != "bert":
            raise ValueError("BERT encoding requires method='bert'")
        
        encoding = self.bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return encoding


# Example usage
if __name__ == "__main__":
    # Word tokenization
    word_tokenizer = Tokenizer(method="word")
    text = "This is a great product! I love it."
    tokens = word_tokenizer.tokenize(text)
    print(f"Word Tokens: {tokens}")
    
    # Sentence tokenization
    sent_tokenizer = Tokenizer(method="sentence")
    text = "This is great! I love it. Will buy again."
    sentences = sent_tokenizer.tokenize(text)
    print(f"Sentences: {sentences}")
    
    # BERT tokenization
    bert_tokenizer = Tokenizer(method="bert")
    text = "This product is amazing!"
    bert_tokens = bert_tokenizer.tokenize(text)
    print(f"BERT Tokens: {bert_tokens}")