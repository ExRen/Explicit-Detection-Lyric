"""
model.py - Model loading and management utilities

This module handles the loading and initialization of BERT models for explicit content detection.
It provides functions to load pretrained models from checkpoints and prepare them for inference.
"""

import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW

class ModelLoader:
    """
    Class responsible for loading and initializing BERT models for explicit content detection.
    """
    
    @staticmethod
    def load_checkpoint(checkpoint_path, map_location=None):
        """
        Load model checkpoint from a .pt file
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            map_location (torch.device, optional): Device to load the model to
            
        Returns:
            tuple: (model, checkpoint, device) - Loaded model, checkpoint data, and device
        """
        # Determine device for loading
        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load checkpoint
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Initialize model
            print("Initializing BERT model...")
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            
            # Load model state with strict=False to ignore missing keys
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Model state loaded successfully (with some keys ignored)")
            else:
                print("No model_state_dict found in checkpoint, using base model")
            
            # Load to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            print(f"Model loaded to {device}")
            
            # Create or update evaluation metrics
            if 'evaluation' not in checkpoint:
                checkpoint['evaluation'] = {}
                
                # Extract accuracy metrics from checkpoint
                if 'val_accuracy' in checkpoint:
                    checkpoint['evaluation']['Validation Accuracy'] = checkpoint['val_accuracy']
                
                if 'test_accuracy' in checkpoint:
                    checkpoint['evaluation']['Test Accuracy'] = checkpoint['test_accuracy']
                    
                if 'optimal_threshold' in checkpoint.get('params', {}):
                    checkpoint['evaluation']['Optimal Threshold'] = checkpoint['params']['optimal_threshold']
                
                # Add metrics from classification_report if available
                if 'classification_report' in checkpoint and isinstance(checkpoint['classification_report'], dict):
                    report = checkpoint['classification_report']
                    
                    if 'accuracy' in report:
                        checkpoint['evaluation']['Overall Accuracy'] = report['accuracy']
                    
                    # Add macro avg metrics
                    if 'macro avg' in report:
                        macro = report['macro avg']
                        if 'precision' in macro:
                            checkpoint['evaluation']['Macro Precision'] = macro['precision']
                        if 'recall' in macro:
                            checkpoint['evaluation']['Macro Recall'] = macro['recall']
                        if 'f1-score' in macro:
                            checkpoint['evaluation']['Macro F1-Score'] = macro['f1-score']
            
            # Ensure we have a basic structure even if checkpoint is minimal
            if not checkpoint['evaluation']:
                checkpoint['evaluation'] = {'Accuracy': 0.0}
                
            return model, checkpoint, device
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Loading base model as fallback...")
            
            # Create a basic model as fallback
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Create placeholder checkpoint
            checkpoint = {
                'params': {'learning_rate': 2e-5, 'batch_size': 16},
                'evaluation': {'accuracy': 0.0}
            }
            
            print(f"Fallback model loaded to {device}")
            return model, checkpoint, device