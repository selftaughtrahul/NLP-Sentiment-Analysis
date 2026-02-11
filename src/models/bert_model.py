"""
BERT Sentiment Classifier
Fine-tuned BERT for sentiment analysis
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils.config import MODEL_CONFIGS, MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SentimentDataset(Dataset):
    """Dataset for BERT"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTSentimentClassifier(nn.Module):
    """BERT-based sentiment classifier"""
    
    def __init__(self, n_classes=3, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


class BERTTrainer:
    """Trainer for BERT model"""
    
    def __init__(self, model_name='bert-base-uncased', n_classes=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BERTSentimentClassifier(n_classes=n_classes)
        self.model.to(self.device)
        
        self.config = MODEL_CONFIGS['bert']
        logger.info(f"Initialized BERT on {self.device}")
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """
        Train BERT model
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
        """
        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, 
            self.tokenizer, 
            self.config['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        total_steps = len(train_loader) * self.config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_loss:.4f}")
            
            # Validation
            if val_texts is not None:
                val_acc = self.evaluate(val_texts, val_labels)
                logger.info(f"Validation accuracy: {val_acc:.4f}")
    
    def predict(self, texts):
        """Predict sentiment for texts"""
        self.model.eval()
        
        dataset = SentimentDataset(
            texts, [0] * len(texts),  # Dummy labels
            self.tokenizer,
            self.config['max_length']
        )
        
        loader = DataLoader(dataset, batch_size=self.config['batch_size'])
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts):
        """Predict probabilities"""
        self.model.eval()
        
        dataset = SentimentDataset(
            texts, [0] * len(texts),
            self.tokenizer,
            self.config['max_length']
        )
        
        loader = DataLoader(dataset, batch_size=self.config['batch_size'])
        
        probabilities = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, texts, labels):
        """Evaluate model"""
        predictions = self.predict(texts)
        accuracy = (predictions == labels).mean()
        return accuracy
    
    def save(self, filepath: Path = None):
        """Save model"""
        if filepath is None:
            filepath = MODELS_DIR / "bert_model.pth"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, filepath)
        
        # Save tokenizer
        tokenizer_path = filepath.parent / "bert_tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Saved BERT model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None):
        """Load model"""
        if filepath is None:
            filepath = MODELS_DIR / "bert_model.pth"
        
        trainer = cls()
        
        checkpoint = torch.load(filepath, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        tokenizer_path = filepath.parent / "bert_tokenizer"
        trainer.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        logger.info(f"Loaded BERT model from {filepath}")
        return trainer