"""
visualizer.py - Visualization utilities

This module provides functions to visualize classification results and model predictions,
including charts, graphs, and distribution plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

class Visualizer:
    """
    Class that provides visualization utilities for classification results
    and model predictions.
    """
    
    @staticmethod
    def visualize_class_distribution(df):
        """
        Create a visualization of class distribution from classification results
        
        Args:
            df (pandas.DataFrame): DataFrame containing classification results with 'is_explicit' column
            
        Returns:
            matplotlib.figure.Figure: Figure containing the class distribution chart
        """
        fig, ax = plt.subplots(figsize=(6, 3))
        explict_count = df['is_explicit'].value_counts()
        sns.barplot(x=explict_count.index.map({True: 'Eksplisit', False: 'Non-Eksplisit'}), 
                    y=explict_count.values, ax=ax)
        ax.set_title('Distribusi Kelas Hasil Klasifikasi')
        ax.set_ylabel('Jumlah Lagu')
        plt.tight_layout()
        return fig

    @staticmethod
    def create_confidence_bars(categories, values, colors):
        """
        Create a bar chart to visualize confidence scores
        
        Args:
            categories (list): List of category names
            values (list): List of confidence values
            colors (list): List of colors for each bar
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with confidence score visualization
        """
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            yaxis=dict(range=[0, 100], title="Probabilitas (%)"),
            margin=dict(t=30, b=30, l=30, r=30),
            height=400
        )
        
        return fig

    @staticmethod
    def create_confidence_bars_matplotlib(categories, values, colors):
        """
        Create a matplotlib bar chart to visualize confidence scores
        
        Args:
            categories (list): List of category names
            values (list): List of confidence values
            colors (list): List of colors for each bar
            
        Returns:
            matplotlib.figure.Figure: Figure containing the confidence score chart
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        bars = ax.bar(categories, values, color=colors)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probabilitas (%)')
        plt.tight_layout()
        
        return fig

    @staticmethod
    def create_confusion_matrix(cm):
        """
        Create a visualization of confusion matrix
        
        Args:
            cm (numpy.ndarray): Confusion matrix from sklearn.metrics.confusion_matrix
            
        Returns:
            matplotlib.figure.Figure: Figure containing the confusion matrix visualization
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['False', 'True'],
            yticklabels=['False', 'True'],
            ax=ax,
            annot_kws={'size': 7}
        )
        
        # Set labels
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)  # Ukuran font label sumbu X
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)  # Ukuran font label sumbu Y
        ax.set_xlabel('Prediksi', fontsize=9)
        ax.set_ylabel('Aktual', fontsize=9)   
        ax.set_title('Confusion Matrix', fontsize=9)
        
        # Calculate and display metrics on the plot
        tn, fp, fn, tp = cm.ravel()
        total = np.sum(cm)
        
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) 
        recall = tp / (tp + fn) 
        f1 = 2 * precision * recall / (precision + recall)
        
        stats_text = f"Accuracy: {accuracy:.4f}\n"
        stats_text += f"Precision: {precision:.4f}\n"
        stats_text += f"Recall: {recall:.4f}\n"
        stats_text += f"F1-Score: {f1:.4f}"
        
        # Add a text box with metrics
        plt.figtext(0.01, 0.01, stats_text, wrap=True, fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig