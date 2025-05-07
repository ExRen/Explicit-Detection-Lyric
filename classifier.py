"""
classifier.py - Text classification functionality

This module handles the prediction and classification of text content,
providing both single text and batch processing capabilities.
"""

import sys
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: NumPy could not be imported. Some functionalities will be limited.")
    NUMPY_AVAILABLE = False

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer import Tokenize

class TextClassifier:
    """
    Class responsible for classifying text content, providing both single text
    and batch processing capabilities.
    """
    
    @staticmethod
    def predict_text(model, tokenizer, texts, device):
        """
        Predict whether texts contain explicit content with confidence scores
        
        Args:
            model: The BERT classification model
            tokenizer: BERT tokenizer instance
            texts (list): List of text strings to classify
            device (torch.device): Device to run inference on
            
        Returns:
            tuple: (predictions, confidence_scores) - Boolean predictions and confidence scores
        """
        # Tokenize input texts
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Ensure logits are in the correct format
            logits = outputs.logits.float()
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Convert to Python lists
        predictions = predictions.cpu().numpy()
        confidence_scores = probs.cpu().numpy()
        
        # Add logging for debugging
        print(f"Raw predictions: {predictions}")
        print(f"Raw confidence scores: {confidence_scores}")
        
        return predictions, confidence_scores

    @staticmethod
    def process_batch_data(model, tokenizer, texts, device, batch_size=16):
        """
        Process a large batch of texts efficiently using batched inference
        
        Args:
            model: The BERT classification model
            tokenizer: BERT tokenizer instance
            texts (list): List of text strings to classify
            device (torch.device): Device to run inference on
            batch_size (int): Batch size for processing
            
        Returns:
            tuple: (predictions, confidence_scores) - Boolean predictions and confidence scores
        """
        # Check if NumPy is available
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for batch processing but isn't available. Please install NumPy: pip install numpy")
            
        # Tokenize all texts
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        # Create dataset and dataloader
        dataset = Tokenize(encodings)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Predict batch by batch
        all_predictions = []
        all_confidence = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_confidence.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_confidence)