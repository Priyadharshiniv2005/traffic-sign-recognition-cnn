import os
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 5
IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 32
EPOCHS = 15

CLASS_NAMES = ['Stop', 'No_Entry', 'Speed_Limit_50', 'Yield', 'Pedestrian_Crossing']


def build_custom_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    """Build simple custom CNN for 5 traffic sign classes."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_data(data_dir='data/train'):
    images = []
    labels = []
    for idx, cname in enumerate(CLASS_NAMES):
        cdir = os.path.join(data_dir, cname)
        if not os.path.exists(cdir):
            print(f'Warning: missing {cdir}', file=sys.stderr)
            continue
        for fname in os.listdir(cdir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp')):
                fpath = os.path.join(cdir, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(idx)
    return np.array(images), np.array(labels)


def main():
    try:
        os.makedirs('models', exist_ok=True)
        print('Loading data from data/train...')
        X, y = load_data('data/train')
        print(f'Loaded {len(X)} images')
        if len(X) == 0:
            raise FileNotFoundError('No images found in data/train. Ensure class folders exist with images.')
        y_cat = to_categorical(y, num_classes=NUM_CLASSES)
        X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
        model = build_custom_cnn()
        model.summary()
        print('\nTraining...')
        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
        model_path = os.path.join('models', 'traffic_sign_cnn.keras')
        model.save(model_path)
        print(f'Model saved to {model_path}')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

