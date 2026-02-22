# Hand Gesture Recognition using CNN (Simple Version)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# 1. Image size
IMG_SIZE = 64

# 2. Dataset path
DATA_DIR = "data"

# 3. Load images
X = []
y = []
class_names = []

for idx, folder in enumerate(os.listdir(DATA_DIR)):
    class_names.append(folder)
    folder_path = os.path.join(DATA_DIR, folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(idx)

X = np.array(X) / 255.0
y = to_categorical(y)


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])


# 6. Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# 7. Train model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))


# 8. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)