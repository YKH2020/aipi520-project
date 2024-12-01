"""
Random Forest model for gender classification project.
Includes both audio-only and full-feature implementations from original notebook.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from typing import Tuple, Dict, Any, Optional

class GenderRandomForest:
    """
    Random Forest classifier for gender prediction.
    Implements both audio-feature and all-feature versions.
    """
    
    def __init__(self, n_estimators: int = 200):
        """
        Initialize model with parameters from original script.
        
        Args:
            n_estimators: Number of trees in forest
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.features_columns = ['sentence', 'accents', 'votes_diff', 'age_enc']
    
    def train_audio_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model using only audio features.
        Direct implementation from original script.
        
        Args:
            df: DataFrame with 'features' and 'gen_enc' columns
            
        Returns:
            Dictionary containing training results and data splits
        """
        # Prepare features and targets
        X = np.vstack(df['features'])
        y = df['gen_enc']
        
        # Train-test split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_val_pred = self.model.predict(X_val)
        
        # Print metrics as in original script
        print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        print("Classification Report:\n", classification_report(y_val, y_val_pred))
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'accuracy': accuracy_score(y_val, y_val_pred)
        }
    
    def train_all_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model using all available features.
        Direct implementation from original script.
        
        Args:
            df: DataFrame containing all feature columns
            
        Returns:
            Dictionary containing training results and data splits
        """
        # Copy the relevant columns
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
        
        # Print feature shapes as in original script
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Print metrics as in original script
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'accuracy': accuracy_score(y_test, y_pred)
        }
    
    def predict_single(self, features: np.ndarray, feature_type: str = 'audio') -> int:
        """
        Predict gender for a single sample.
        
        Args:
            features: Feature vector
            feature_type: Type of features ('audio' or 'all')
            
        Returns:
            Predicted class (0 for female, 1 for male)
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if not hasattr(self.model, 'classes_'):
            raise ValueError("Model must be trained before prediction")
        
        # Reshape for single sample prediction
        features = features.reshape(1, -1)
        
        # For all features, apply same preprocessing as training
        if feature_type == 'all':
            if 'votes_diff' in self.scaler.feature_names_in_:
                features = self.scaler.transform(features)
        
        return self.model.predict(features)[0]
    
    def predict_proba_single(self, features: np.ndarray, 
                           feature_type: str = 'audio') -> np.ndarray:
        """
        Get prediction probabilities for a single sample.
        
        Args:
            features: Feature vector
            feature_type: Type of features ('audio' or 'all')
            
        Returns:
            Array of probabilities for each class
        """
        if not hasattr(self.model, 'classes_'):
            raise ValueError("Model must be trained before prediction")
        
        features = features.reshape(1, -1)
        if feature_type == 'all':
            if 'votes_diff' in self.scaler.feature_names_in_:
                features = self.scaler.transform(features)
        
        return self.model.predict_proba(features)[0]
    
    def save_model(self, filepath: str):
        """
        Save model and preprocessing objects.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'features_columns': self.features_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """
        Load saved model and preprocessing objects.
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.features_columns = model_data['features_columns']