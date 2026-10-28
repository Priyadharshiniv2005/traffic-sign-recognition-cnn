# Traffic Sign Recognition Using Convolutional Neural Networks (CNN)

This is an academic project that uses a custom CNN to recognize German traffic signs (GTSRB dataset, 43 classes). It includes data preprocessing, model training/evaluation, and a Streamlit web app for real-time prediction.

## Features
- Custom CNN built from scratch (TensorFlow/Keras)
- GTSRB dataset with 43 classes
- Image preprocessing, normalization, and data augmentation
- Training/validation with accuracy/loss curves
- Confusion matrix and classification report
- Model saving in Keras format
- Streamlit app for image upload, prediction, and confidence score
- Error handling for missing files/images

## Project Structure
`
traffic sign/
├── data/           # GTSRB dataset (not included)
├── models/         # Saved trained models
├── outputs/        # Training graphs, confusion matrix, reports
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── visualization.py
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
└── README.md
`

## Prerequisites
- Python 3.11 or 3.12 (recommended). **Python 3.14 is not currently supported by stable TensorFlow releases.** Use a separate virtual environment.
- pip

## Setup (Windows, VS Code)

### 1) Create virtual environment (recommended)
Using py launcher:
`ash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
`
Or with Python 3.11:
`ash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
`

### 2) Install dependencies
`ash
pip install --upgrade pip
pip install -r requirements.txt
`

### 3) Prepare dataset
1. Download GTSRB dataset (official/source). 
2. Extract and place into data/ in a structure that 	rain.py expects (see comments in 	rain.py). 
3. Do **not** commit dataset files.

### 4) Train model
`ash
python train.py
`
This will validate dataset presence, build a custom CNN, train with validation, save model to models/, and save plots to outputs/.

### 5) Evaluate model
`ash
python evaluate.py
`
Generates confusion matrix, classification report, and plots in outputs/.

### 6) Run Streamlit app
`ash
streamlit run app.py
`
Upload a traffic sign image to get predicted class and confidence score.

## Notes
- No dataset or trained models are included in this repo. You must provide/download them.
- Do not start training until dataset is available and validated.
- Code uses relative paths and includes error handling.
