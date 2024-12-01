# Gender Classification from Speaker Data

## Overview
This project predicts gender classification from audio speaker data using both traditional machine learning (Random Forest) and deep learning models. It processes audio files from the Common Voice dataset to extract features and build models for predicting the gender of the speaker

## Models
* Random Forest with audio features
* Random Forest with all features
* CNN using raw audio
* Dense neural network using extracted features

## Installation
```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Dataset Structure
The project uses Common Voice dataset in the following structure:
```
cv-corpus-10.0-delta-2022-07-04/
├── en/
│   ├── clips/
│   │   └── *.mp3
│   ├── validated.tsv
│   ├── invalidated.tsv
│   ├── other.tsv
│   ├── reported.tsv
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
```

## Training


## Project Structure
```
gender_classification/
├── src/
│   ├── data_loading.py
│   ├── audio_processor.py
│   ├── rf_model.py
│   └── deep_learning.py
├── scripts/
│   ├── training.py
│   └── predictions.py
├── requirements.txt
└── README.md
```

## Model Performance
[Update]