"""
Data processing module for gender classification project.
Handles loading and preprocessing of Common Voice dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Union
import os


class DataLoader:
    """DataLoader class that exactly matches original script functionality."""
    
    def __init__(self, base_dir: str = './cv-corpus-10.0-delta-2022-07-04/en'):
        """
        Initialize DataLoader with path checking.
        
        Args:
            base_dir: Path to CommonVoice dataset directory
            
        Raises:
            FileNotFoundError: If directory or required files don't exist
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {base_dir}")
        
        required_files = ['validated.tsv', 'invalidated.tsv', 'other.tsv', 
                         'reported.tsv', 'dev.tsv', 'train.tsv', 'test.tsv']
        missing_files = [f for f in required_files 
                        if not (self.base_dir / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")

    def _load_and_clean_file(self, filename: str) -> pd.DataFrame:
        """
        Load and clean a single TSV file.
        
        Args:
            filename: Name of file to load
            
        Returns:
            Cleaned DataFrame
        """
        df = pd.read_csv(self.base_dir / filename, delimiter='\t')
        print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')
        
        df_clean = df.drop(labels=['segment'], axis=1)
        print(f'num of NaN rows count: {df_clean.isnull().any(axis=1).sum()}')
        
        df_clean = df_clean.dropna()
        print(f'Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}')
        
        return df_clean

    def load_all_datasets(self) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        Load all datasets following original script.
        
        Returns:
            Dictionary containing all loaded and cleaned datasets
        """
        datasets = {}
        
        # Load validated and split into NaN/non-NaN
        validated = pd.read_csv(self.base_dir / 'validated.tsv', delimiter='\t')
        val = validated.drop(labels=['segment'], axis=1)
        df_with_nans = val[val.isna().any(axis=1)]
        df_without_nans = val[val.notna().all(axis=1)]
        datasets['validated'] = (
            df_with_nans[['client_id', 'path']],
            df_without_nans[['client_id', 'path']]
        )
        
        # Load other datasets
        for filename in ['invalidated', 'other', 'reported', 'dev', 'train', 'test']:
            datasets[filename] = self._load_and_clean_file(f'{filename}.tsv')
        
        return datasets

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get processed training and test data.
        
        Returns:
            Tuple of (training_data, test_data) ready for model training
        """
        # Load and combine train/dev
        train_clean = self._load_and_clean_file('train.tsv')
        dev_clean = self._load_and_clean_file('dev.tsv')
        clean_dev_train = pd.concat([train_clean, dev_clean], ignore_index=True)
        test_clean = self._load_and_clean_file('test.tsv')
        
        # Process votes and drop columns
        def process_votes(df):
            df = df.copy()
            df['votes_diff'] = df['up_votes'] - df['down_votes']
            return df.drop(columns=['up_votes', 'down_votes', 'locale'])
            
        clean_dev_train = process_votes(clean_dev_train)
        test_clean = process_votes(test_clean)
        
        # Check accent overlap
        unique_test = test_clean['accents'].unique()
        unique_train_dev = clean_dev_train['accents'].unique()
        overlap = len(set(unique_test) & set(unique_train_dev)) / len(set(unique_test)) * 100
        print(f"Percentage of overlap between test and train+dev: {overlap:.2f}%")
        
        # Process gender
        def encode_gender(df):
            df = df.copy()
            df = df[df['gender'] != 'other']
            df['gen_enc'] = df['gender'].replace({'male': 1, 'female': 0})
            return df.drop(columns='gender')
            
        clean_dev_train = encode_gender(clean_dev_train)
        test_clean = encode_gender(test_clean)
        
        # Process age
        encoder = LabelEncoder()
        encoder.fit(test_clean['age'])
        
        clean_dev_train['age_enc'] = encoder.transform(clean_dev_train['age'])
        test_clean['age_enc'] = encoder.transform(test_clean['age'])
        
        clean_dev_train = clean_dev_train.drop(columns='age')
        test_clean = test_clean.drop(columns='age')
        
        return clean_dev_train, test_clean

    def save_processed_data(self, train_data: pd.DataFrame, 
                          test_data: pd.DataFrame, output_dir: str = '.'):
        """
        Save processed datasets to files.
        
        Args:
            train_data: Processed training data
            test_data: Processed test data
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_data.to_csv(output_dir / 'processed_train.tsv', sep='\t', index=False)
        test_data.to_csv(output_dir / 'processed_test.tsv', sep='\t', index=False)

    def load_processed_data(self, data_dir: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously processed datasets.
        
        Args:
            data_dir: Directory containing processed data files
            
        Returns:
            Tuple of (train_data, test_data)
        """
        data_dir = Path(data_dir)
        train_data = pd.read_csv(data_dir / 'processed_train.tsv', sep='\t')
        test_data = pd.read_csv(data_dir / 'processed_test.tsv', sep='\t')
        return train_data, test_data