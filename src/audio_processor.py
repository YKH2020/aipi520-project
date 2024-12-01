"""
Audio processing and feature extraction for gender classification project.
Handles audio loading, processing, and feature extraction.
"""

import numpy as np
import librosa
from pathlib import Path
from scipy.signal import find_peaks
from scipy.fftpack import fft
import pandas as pd
from typing import Tuple, Optional, List
import pickle

class AudioProcessor:
    """Processes audio files and extracts features for gender classification."""
    
    def __init__(self, audio_dir: str = './cv-corpus-10.0-delta-2022-07-04/en/clips',
                 max_length: int = 160000):
        """
        Initialize audio processor.
        
        Args:
            audio_dir: Directory containing audio clips
            max_length: Maximum length for audio padding/truncating (for CNN)
            
        Raises:
            FileNotFoundError: If audio directory doesn't exist
        """
        self.audio_dir = Path(audio_dir)
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
            
        self.max_length = max_length
        
        # Run initial test to verify audio loading works
        self.test_audio()
    
    def test_audio(self):
        """Test audio loading with known test file."""
        test_path = self.audio_dir / "common_voice_en_32233439.mp3"
        print(f"Testing file: {test_path}")
        try:
            audio_data, sample_rate = librosa.load(test_path, sr=None)
            print(f"Loaded successfully: {audio_data.shape}, Sample rate: {sample_rate}")
        except Exception as e:
            print(f"Error loading WAV: {e}")
            raise

    def process_audio(self, mp3_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Process a single audio file.
        Exact implementation from original script.
        
        Args:
            mp3_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if failed
        """
        try:
            mp3_file = self.audio_dir / mp3_path
            audio_data, sample_rate = librosa.load(mp3_file, sr=None)
            print(f"Processed {mp3_file}: {audio_data.shape}, Sample rate: {sample_rate}")
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error processing {mp3_path}: {e}")
            return None, None

    def process_datasets(self, train_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process audio files for both datasets.
        
        Args:
            train_df: Training data with 'path' column
            test_df: Test data with 'path' column
            
        Returns:
            Processed dataframes with audio data and features
        """
        # Copy dataframes
        clean_dev_train_audio = train_df.copy()
        test_clean_audio = test_df.copy()

        # Process audio files
        clean_dev_train_audio['audio_data'], clean_dev_train_audio['sample_rate'] = zip(
            *clean_dev_train_audio['path'].apply(self.process_audio)
        )
        test_clean_audio['audio_data'], test_clean_audio['sample_rate'] = zip(
            *test_clean_audio['path'].apply(self.process_audio)
        )

        # Drop path columns
        clean_dev_train_audio = clean_dev_train_audio.drop('path', axis=1)
        test_clean_audio = test_clean_audio.drop('path', axis=1)

        # Extract features
        self.extract_features_for_datasets(clean_dev_train_audio, test_clean_audio)

        # Save processed data
        clean_dev_train_audio.to_csv('clean_dev_train_audio.tsv', sep='\t', index=False)
        test_clean_audio.to_csv('test_clean_audio.tsv', sep='\t', index=False)

        return clean_dev_train_audio, test_clean_audio

    def extract_features_numpy(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract features from audio data.
        Exact implementation from original script.
        
        Args:
            audio_data: Raw audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Array of extracted features
        """
        # Ensure audio data is a NumPy array
        audio_np = np.array(audio_data, dtype=np.float32)
        n = len(audio_np)
        
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(y=audio_np, sr=sample_rate, n_mfcc=13).mean(axis=1)
        
        # 2. Zero-Crossing Rate (ZCR)
        zcr = np.mean(audio_np[:-1] * audio_np[1:] < 0)
        
        # 3. Spectral Centroid
        magnitudes = np.abs(fft(audio_np))[:n // 2]
        frequencies = np.linspace(0, sample_rate / 2, len(magnitudes))
        spectral_centroid = np.sum(frequencies * magnitudes) / (np.sum(magnitudes) + 1e-7)
        
        return np.hstack([mfccs, zcr, spectral_centroid])

    def extract_features_for_datasets(self, train_df: pd.DataFrame, 
                                    test_df: pd.DataFrame) -> None:
        """
        Extract features for both datasets.
        
        Args:
            train_df: Training data with audio_data column
            test_df: Test data with audio_data column
        """
        # Extract and display first feature vector
        features = self.extract_features_numpy(
            train_df['audio_data'].iloc[0], 
            train_df['sample_rate'].iloc[0]
        )
        print(f"Feature vector shape: {features.shape}")
        print(f"Extracted Features: {features}")

        # Extract features for both datasets
        train_df['features'] = train_df.apply(
            lambda row: self.extract_features_numpy(row['audio_data'], row['sample_rate']), 
            axis=1
        )
        test_df['features'] = test_df.apply(
            lambda row: self.extract_features_numpy(row['audio_data'], row['sample_rate']), 
            axis=1
        )

    def preprocess_audio_for_cnn(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for CNN model.
        Exact implementation from original script.
        
        Args:
            audio_data: Raw audio signal
            
        Returns:
            Processed audio data
        """
        audio_np = np.array(audio_data, dtype=np.float32)
        # Normalize
        audio_np = (audio_np - np.mean(audio_np)) / (np.std(audio_np) + 1e-7)
        # Pad or truncate
        if len(audio_np) < self.max_length:
            audio_np = np.pad(audio_np, (0, self.max_length - len(audio_np)), mode='constant')
        else:
            audio_np = audio_np[:self.max_length]
        return audio_np

    def save_features(self, features: dict, filepath: str):
        """
        Save extracted features to file.
        
        Args:
            features: Dictionary of features
            filepath: Path to save features
        """
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)

    def load_features(self, filepath: str) -> dict:
        """
        Load previously extracted features.
        
        Args:
            filepath: Path to features file
            
        Returns:
            Dictionary of features
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)