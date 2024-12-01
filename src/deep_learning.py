"""
Deep learning models for gender classification project.
Implements both CNN and Dense architectures from original script.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

class GenderDeepLearning:
    """
    Deep learning models for gender classification.
    Includes both CNN (for audio) and Dense (for all features) implementations.
    """
    
    def __init__(self, max_length: int = 160000):
        """
        Initialize models with parameters from original script.
        
        Args:
            max_length: Maximum length for audio padding/truncating
        """
        self.max_length = max_length
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.features_columns = ['sentence', 'accents', 'votes_diff', 'age_enc']
        self.history = None
    
    def preprocess_audio_for_cnn(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for CNN input.
        Direct implementation from original script.
        
        Args:
            audio_data: Raw audio signal
            
        Returns:
            Preprocessed audio data
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
    
    def build_cnn_model(self):
        """Build CNN model with exact architecture from original script."""
        self.model = tf.keras.models.Sequential([
            # First CNN block
            tf.keras.layers.Conv1D(16, kernel_size=5, strides=2, padding='same',
                                 activation='relu', input_shape=(self.max_length, 1)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            # Second CNN block
            tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, padding='same',
                                 activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            # Third CNN block
            tf.keras.layers.Conv1D(64, kernel_size=5, strides=2, padding='same',
                                 activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
    
    def build_dense_model(self, input_shape: int):
        """
        Build dense model with exact architecture from original script.
        
        Args:
            input_shape: Number of input features
        """
        self.model = tf.keras.models.Sequential([
            # First block
            tf.keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            
            # Second block
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Third block
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Fourth block
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Output layer
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
    
    def train_cnn_model(self, df: pd.DataFrame, epochs: int = 10, 
                       batch_size: int = 32) -> Dict[str, Any]:
        """
        Train CNN model using audio data.
        Direct implementation from original script.
        
        Args:
            df: DataFrame with audio_data column
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training results
        """
        # Preprocess audio data
        df = df.copy()
        df['audio_data'] = df['audio_data'].apply(self.preprocess_audio_for_cnn)
        
        # Prepare features and labels
        X = np.vstack(df['audio_data'].values).reshape(-1, self.max_length, 1)
        y = df['gen_enc'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        
        # Build and train model
        self.build_cnn_model()
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate
        metrics = self._evaluate_model(X_test, y_test)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'history': self.history.history,
            'metrics': metrics
        }
    
    def train_dense_model(self, df: pd.DataFrame, epochs: int = 10, 
                         batch_size: int = 32) -> Dict[str, Any]:
        """
        Train dense model using all features.
        Direct implementation from original script.
        
        Args:
            df: DataFrame with all feature columns
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training results
        """
        # Prepare features
        feature_df = df[self.features_columns].copy()
        
        # Encode categorical columns
        for col in ['sentence', 'accents']:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col])
            self.label_encoders[col] = le
            
        # Standardize numerical columns
        feature_df[['votes_diff']] = self.scaler.fit_transform(feature_df[['votes_diff']])
        
        # Prepare features and labels
        X = feature_df.values
        y = df['gen_enc'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Build and train model
        self.build_dense_model(X_train.shape[1])
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate
        metrics = self._evaluate_model(X_test, y_test)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'history': self.history.history,
            'metrics': metrics
        }
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_probs
        }
    
    def save_model(self, filepath: str):
        """
        Save model and preprocessing objects.
        
        Args:
            filepath: Path to save model
        """
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save preprocessors
        preprocessing = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'max_length': self.max_length,
            'features_columns': self.features_columns
        }
        with open(filepath + '.prep', 'wb') as f:
            pickle.dump(preprocessing, f)
    
    def load_model(self, filepath: str):
        """
        Load saved model and preprocessing objects.
        
        Args:
            filepath: Path to model file
        """
        # Load Keras model
        self.model = tf.keras.models.load_model(filepath)
        
        # Load preprocessors
        with open(filepath + '.prep', 'rb') as f:
            preprocessing = pickle.dump(f)
            self.label_encoders = preprocessing['label_encoders']
            self.scaler = preprocessing['scaler']
            self.max_length = preprocessing['max_length']
            self.features_columns = preprocessing['features_columns']