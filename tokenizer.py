"""
tokenizer.py - Tokenization utilities

This module handles text tokenization and dataset preparation for BERT models.
It includes functions to convert raw text into tokenized inputs suitable for model inference.
"""

from transformers import BertTokenizer
from torch.utils.data import Dataset

class Tokenize(Dataset):
    """
    PyTorch Dataset for handling lyrics data
    
    This dataset handles the tokenized encodings and optional labels for lyrics data,
    making it compatible with PyTorch DataLoader for batch processing.
    """
    def __init__(self, encodings, labels=None):
        """
        Initialize the dataset with encodings and optional labels
        
        Args:
            encodings (dict): The tokenizer encodings with 'input_ids' and 'attention_mask'
            labels (list, optional): Optional labels for supervised learning
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            dict: A dictionary containing the input tensors and optionally the label
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

class TokenizerManager:
    """
    Class responsible for managing tokenization operations and dataset preparation
    for BERT models.
    """
    
    @staticmethod
    def get_tokenizer():
        """
        Get the BERT tokenizer for processing lyric texts
        
        Returns:
            BertTokenizer: An instance of the BERT tokenizer
        """
        return BertTokenizer.from_pretrained('bert-base-uncased')

    @staticmethod
    def clean_texts(text_series):
        """
        Clean and prepare texts for tokenization
        
        Args:
            text_series (list or Series): A series or list of text strings to clean
            
        Returns:
            list: List of cleaned text strings
        """
        return [str(text) if text is not None else "" for text in text_series]