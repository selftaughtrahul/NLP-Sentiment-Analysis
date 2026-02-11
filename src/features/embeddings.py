"""
Word Embeddings Module
Handles Word2Vec, GloVe, and BERT embeddings
"""
import numpy as np
from typing import List, Dict
import torch
from transformers import BertModel, BertTokenizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BERTEmbeddings:
    """Extract BERT embeddings for text"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize BERT model
        
        Args:
            model_name: Pretrained BERT model name
        """
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"BERT model loaded on {self.device}")
    
    def get_embedding(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Get BERT embedding for single text
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Embedding vector (768-dim for base BERT)
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return cls_embedding.squeeze()
    
    def get_embeddings_batch(self, texts: List[str], 
                            max_length: int = 512,
                            batch_size: int = 16) -> np.ndarray:
        """
        Get BERT embeddings for multiple texts
        
        Args:
            texts: List of texts
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_texts, 768)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Use [CLS] token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


# Example usage
if __name__ == "__main__":
    # Test BERT embeddings
    bert = BERTEmbeddings()
    
    text = "This is a great product!"
    embedding = bert.get_embedding(text)
    
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10]}")
    
    # Batch processing
    texts = [
        "I love this product",
        "Terrible service",
        "Average quality"
    ]
    
    embeddings = bert.get_embeddings_batch(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")