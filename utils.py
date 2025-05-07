"""
utils.py - General utility functions

This module provides general utilities for file handling, data processing,
and other helper functions used across the application.
"""

import base64
import os

class ApplicationUtils:
    """
    Class providing general utility functions for file handling, data processing,
    and other helper operations used across the application.
    """
    
    @staticmethod
    def get_csv_download_link(df, filename="hasil_klasifikasi.csv"):
        """
        Generate a download link for a DataFrame as CSV
        
        Args:
            df (pandas.DataFrame): DataFrame to convert to CSV
            filename (str): Name of the file to download
            
        Returns:
            str: HTML link for downloading the CSV file
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download hasil klasifikasi</a>'
        return href

    @staticmethod
    def get_model_files(checkpoint_folder):
        """
        Get a list of model checkpoint files from a folder
        
        Args:
            checkpoint_folder (str): Path to the folder containing model checkpoints
            
        Returns:
            list: List of model checkpoint filenames
        """
        if not os.path.exists(checkpoint_folder):
            return []
        
        model_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pt')]
        return model_files