import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

from utils.visualization import plot_confusion_matrix, save_classification_report

NUM_CLASSES = 43

def check_files(model_path='models/traffic_sign_cnn.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f\"Model not found: {model_path}. Train the model first with train.py\")
    if not os.path.exists('data'):
        raise FileNotFoundError(f\"Data directory not found. Dataset required for evaluation.\")
    return True

def main():
    try:
        check_files()
        print(\"Loading model...\")
        model = load_model('models/traffic_sign_cnn.h5')
        print(\"Model loaded. Note: Evaluation requires test data.\")
        print(\"Implement test data loading based on your dataset structure.\")
        print(\"This template will generate confusion matrix and classification report when test data is provided.\")
    except Exception as e:
        print(f\"Error: {e}\", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
