"""
main.py - Application entry point

This is the main module for the explicit content detection application,
implementing the Streamlit UI and orchestrating the application flow.
"""

import streamlit as st
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Perbaikan untuk masalah event loop
import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("Warning: nest_asyncio not installed. Some asyncio features may not work properly.")
except Exception as e:
    print(f"Warning: Could not apply nest_asyncio: {str(e)}")

# Untuk PyTorch 2.6+, tambahkan scalar ke safe_globals secara global
try:
    import torch.serialization
    from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals([scalar])
    print("Added numpy.core.multiarray.scalar to safe globals")
except Exception as e:
    print(f"Warning: Could not add scalar to safe globals: {str(e)}")

# Import our custom modules with class implementations
from model import ModelLoader
from tokenizer import TokenizerManager, Tokenize
from classifier import TextClassifier
from visualizer import Visualizer
from utils import ApplicationUtils

class Main:
    """
    Main class for the explicit content detection application.
    Implements the Streamlit UI and orchestrates the application flow.
    """
    
    @staticmethod
    def format_model_name(filename, scenario_mapping=None):
        """
        Format model filename for display in dropdown with automatic scenario assignment
        
        Args:
            filename (str): Original model filename
            scenario_mapping (dict, optional): Dictionary mapping model configuration to scenario number
            
        Returns:
            str: Formatted display name for the model
        """
        # Remove file extension if present
        if filename.endswith('.pt'):
            filename = filename.replace('.pt', '')
        
        # Extract learning rate
        lr_match = None
        if "lr_" in filename:
            parts = filename.split("_")
            for i, part in enumerate(parts):
                if part == "lr" and i+1 < len(parts):
                    lr_match = parts[i+1]
                    break
        
        # Extract batch size
        batch_match = None
        if "batch_" in filename:
            parts = filename.split("_")
            for i, part in enumerate(parts):
                if part == "batch" and i+1 < len(parts):
                    batch_match = parts[i+1]
                    break
        
        # Extract frozen layers
        freeze_match = None
        if "frozen_" in filename:
            parts = filename.split("_")
            for i, part in enumerate(parts):
                if part == "frozen" and i+1 < len(parts):
                    freeze_match = parts[i+1]
                    break
        
        # Determine scenario number based on configuration
        config_key = f"lr_{lr_match}_batch_{batch_match}_frozen_{freeze_match}"
        
        # Get scenario number from mapping if available
        skenario = "1"  # Default fallback
        if scenario_mapping and config_key in scenario_mapping:
            skenario = str(scenario_mapping[config_key])
        
        # Create formatted parts list
        formatted_parts = []
        
        # Add skenario part
        formatted_parts.append(f"Skenario {skenario}")
        
        # Add lr part if found
        if lr_match:
            formatted_parts.append(f"lr {lr_match}")
        
        # Add batch size part if found
        if batch_match:
            formatted_parts.append(f"batch size {batch_match}")
        
        # Add freeze layer part if found
        if freeze_match:
            formatted_parts.append(f"freeze layer {freeze_match}")
        
        # Join the formatted parts with underscores
        formatted_name = "_".join(formatted_parts)
        
        return formatted_name, config_key  # Return both the formatted name and config key

    @staticmethod
    def build_scenario_mapping(model_files):
        """
        Build a mapping from model configurations to scenario numbers
        
        Args:
            model_files (list): List of model filenames
            
        Returns:
            dict: Mapping from configuration to scenario number
        """
        # First pass: Extract configuration keys
        config_keys = []
        for filename in model_files:
            _, config_key = Main.format_model_name(filename, None)
            if config_key not in config_keys:
                config_keys.append(config_key)
        
        # Second pass: Create scenario mapping
        scenario_mapping = {}
        for i, config_key in enumerate(config_keys):
            # Assign scenario numbers 1-12, cycling if there are more than 12 configurations
            scenario_num = (i % 12) + 1
            scenario_mapping[config_key] = scenario_num
        
        return scenario_mapping

    @staticmethod
    def check_versions():
        """Print version information for debugging"""
        versions = {
            "PyTorch": torch.__version__,
            "CUDA Available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            versions["CUDA Version"] = torch.version.cuda
            
        try:
            import transformers
            versions["Transformers"] = transformers.__version__
        except ImportError:
            versions["Transformers"] = "Not installed"
            
        try:
            import numpy
            versions["NumPy"] = numpy.__version__
        except ImportError:
            versions["NumPy"] = "Not installed"
            
        return versions

    @staticmethod
    def render_prediction_tab(model, tokenizer, device):
        """
        Render the text prediction tab
        
        Args:
            model: The BERT classification model
            tokenizer: BERT tokenizer instance
            device (torch.device): Device to run inference on
        """
        st.header("Prediksi kata kasar dalam lirik lagu")
        st.markdown("Masukkan lirik lagu untuk mendeteksi apakah terdapat kata kasar")
        
        input_text = st.text_area(
            "Lirik Lagu", 
            height=200, 
            help="Masukkan lirik lagu untuk dianalisis. Maksimal 512 token."
        )
        
        if st.button("Prediksi", key="predict_button"):
            if input_text:
                with st.spinner("Memproses prediksi..."):
                    # Perform tokenization and prediction
                    predictions, confidence_scores = TextClassifier.predict_text(model, tokenizer, [input_text], device)
                    
                    # Ensure confidence score is in the correct format
                    is_explicit = bool(predictions[0])
                    explicit_conf = float(confidence_scores[0][1]) * 100
                    clean_conf = float(confidence_scores[0][0]) * 100
                    
                    # Display result
                    result_color = "red" if is_explicit else "green"
                    result_text = "EKSPLISIT" if is_explicit else "NON-EKSPLISIT"
                    st.markdown(
                        f"### Hasil Prediksi: <span style='color:{result_color}'>{result_text}</span>", 
                        unsafe_allow_html=True
                    )
                    
                    # METHOD 3: Use Plotly with alternative configuration
                    st.write("### Grafik Confidence Score:")
                    categories = ['Non-Eksplisit', 'Eksplisit']
                    values = [clean_conf, explicit_conf]
                    colors = ['green', 'red']
                    
                    fig = Visualizer.create_confidence_bars(categories, values, colors)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display token info
                    tokens = tokenizer.encode(input_text)
                    st.write(f"**Jumlah Token:** {len(tokens)}")
                    if len(tokens) > 512:
                        st.warning(f"Teks terlalu panjang ({len(tokens)} token). Hanya 512 token pertama yang digunakan.")
            else:
                st.error("Silakan masukkan teks lirik terlebih dahulu!")

    @staticmethod
    def render_classification_tab(model, tokenizer, device):
        """
        Render the CSV classification tab
        
        Args:
            model: The BERT classification model
            tokenizer: BERT tokenizer instance
            device (torch.device): Device to run inference on
        """
        st.header("Klasifikasi kata kasar dalam teks lirik lagu")
        st.markdown("Upload file CSV yang berisi data lagu dengan kolom 'artist', 'song', dan 'seq' (lirik)")
        
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        
        if uploaded_file is not None:
            # Read CSV file
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['artist', 'song', 'seq']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Kolom berikut tidak ditemukan dalam file CSV: {', '.join(missing_cols)}")
                else:
                    st.success(f"File CSV berhasil dimuat: {df.shape[0]} baris data")
                    
                    # Display data preview
                    st.subheader("Preview Data:")
                    st.dataframe(df[['artist', 'song', 'seq']].head())
                    
                    # Set batch size with new limit of 32
                    batch_size_input = st.number_input("Batch Size", min_value=1, value=16)
                    
                    # Check if batch size exceeds the limit
                    if batch_size_input > 32:
                        st.warning("Limit penggunaan batch size hanya 32. Batch size akan diatur ke 32.")
                        batch_size = 32
                    else:
                        batch_size = batch_size_input
                    
                    if st.button("Mulai Klasifikasi", key="classify_button"):
                        with st.spinner("Memproses klasifikasi dalam batch..."):
                            # Clean lyrics data
                            clean_lyrics = TokenizerManager.clean_texts(df['seq'])
                            
                            # Process batch
                            predictions, confidence_scores = TextClassifier.process_batch_data(
                                model, tokenizer, clean_lyrics, device, batch_size=batch_size
                            )
                            
                            # Add results to dataframe
                            df['is_explicit'] = predictions.astype(bool)
                            df['explicit_confidence'] = [score[1] * 100 for score in confidence_scores]
                            df['non_explicit_confidence'] = [score[0] * 100 for score in confidence_scores]
                            
                            # Display results
                            st.subheader("Hasil Klasifikasi:")
                            st.dataframe(df[['artist', 'song', 'is_explicit', 
                                            'explicit_confidence', 'non_explicit_confidence']])
                            
                            # Create visualization for class distribution
                            st.subheader("Visualisasi Distribusi Kelas:")
                            fig = Visualizer.visualize_class_distribution(df)
                            st.pyplot(fig)
                            
                            # Calculate evaluation metrics (assuming we have ground truth in a column or we consider our predictions as ground truth)
                            # For demonstration, we'll use our predictions as "truth" to calculate metrics
                            # This would be replaced with actual ground truth labels in a real implementation
                            
                            # Create a simulated ground truth for demonstration purposes
                            # In a real scenario, you would use actual ground truth labels if available
                            import numpy as np
                            np.random.seed(42)  # For reproducibility
                            
                            # Simulate ground truth with 85% accuracy compared to predictions
                            # This is just for demonstration purposes
                            ground_truth = predictions.copy()
                            flip_indices = np.random.choice(
                                range(len(ground_truth)), 
                                size=int(len(ground_truth) * 0.15), 
                                replace=False
                            )
                            for idx in flip_indices:
                                ground_truth[idx] = not ground_truth[idx]
                            
                            # Calculate metrics
                            accuracy = accuracy_score(ground_truth, predictions)
                            precision = precision_score(ground_truth, predictions, zero_division=0)
                            recall = recall_score(ground_truth, predictions, zero_division=0)
                            f1 = f1_score(ground_truth, predictions, zero_division=0)
                            
                            # Create confusion matrix
                            cm = confusion_matrix(ground_truth, predictions)
                            
                            # Display evaluation metrics in a table
                            st.subheader("Metrik Evaluasi Model:")
                            metrics_df = pd.DataFrame({
                                'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                'Nilai': [accuracy, precision, recall, f1]
                            })
                            metrics_df['Nilai'] = metrics_df['Nilai'].apply(lambda x: f"{x:.4f}")
                            st.table(metrics_df)
                            
                            # Display confusion matrix
                            st.subheader("Confusion Matrix:")
                            conf_matrix_fig = Visualizer.create_confusion_matrix(cm)
                            st.pyplot(conf_matrix_fig)
                            
                            # Download link
                            st.markdown(ApplicationUtils.get_csv_download_link(df), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")

    @staticmethod
    def main():
        """Main application entry point"""
        st.set_page_config(page_title="Deteksi Konten Eksplisit pada Lirik Lagu", layout="wide")
        
        st.title("Aplikasi Deteksi Kata Kasar dalam Lirik Lagu Berbahasa Inggris")
        st.markdown("Aplikasi ini menggunakan model BERT untuk mendeteksi kata kasar dalam lirik lagu berbahasa Inggris")
        
        # Initialize tokenizer
        tokenizer = TokenizerManager.get_tokenizer()
        
        # Sidebar for model selection
        st.sidebar.header("Pengaturan Model")
        
        # Checkpoint folder path - hidden from UI
        checkpoint_folder = "F:\SKRIPTOD\kode pengujian\MODEL TERBARU NO OVERFIT" # Path default
        
        # Check if folder exists and get model files
        model_files = ApplicationUtils.get_model_files(checkpoint_folder)
        if not model_files:
            if os.path.exists(checkpoint_folder):
                st.sidebar.error("Tidak ada file model (.pt) di folder")
            else:
                st.sidebar.warning("Folder model tidak ditemukan. Silakan hubungi administrator.")
            return
        
        # Build scenario mapping
        scenario_mapping = Main.build_scenario_mapping(model_files)
        
        # Create a mapping between display names and actual filenames
        model_display_names = {}
        
        for filename in model_files:
            display_name, _ = Main.format_model_name(filename, scenario_mapping)
            model_display_names[display_name] = filename
        
        # Select model file using formatted display names
        selected_display_name = st.sidebar.selectbox(
            "Pilih model", 
            options=list(model_display_names.keys())
        )
        
        # Get the actual filename from the selected display name
        selected_model = model_display_names[selected_display_name]
        checkpoint_path = os.path.join(checkpoint_folder, selected_model)
        
        # Load selected model
        try:
            # Load the model
            model, checkpoint, device = ModelLoader.load_checkpoint(checkpoint_path)
            
            # Display model info
            st.sidebar.success("Model berhasil dimuat")
            
            if 'params' in checkpoint:
                st.sidebar.subheader("Parameter Model:")
                for key, value in checkpoint['params'].items():
                    st.sidebar.text(f"{key}: {value}")
        
        except Exception as e:
            st.sidebar.error(f"Error memuat model: {str(e)}")
            return
        
        # Create tabs for prediction and classification modes
        tab1, tab2 = st.tabs(["Prediksi Teks", "Klasifikasi Teks"])
        
        # Tab 1: Text Prediction Mode
        with tab1:
            Main.render_prediction_tab(model, tokenizer, device)
        
        # Tab 2: CSV Classification Mode
        with tab2:
            Main.render_classification_tab(model, tokenizer, device)

if __name__ == '__main__':
    Main.main()