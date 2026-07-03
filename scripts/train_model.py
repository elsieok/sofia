# scripts/train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf

# tensorflow-metal (Apple Silicon GPU plugin) can hang indefinitely on the
# first model.fit() call. Disabling the GPU forces CPU-only training, which
# is plenty fast for a model this small.
tf.config.set_visible_devices([], 'GPU')

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. Load and Preprocess Data ---
print("Loading and preprocessing data...")

# Load the dataset from CSV (label,x0,y0,z0,...,x20,y20,z20)
df = pd.read_csv('data/kodaly_dataset.csv')

# Separate labels (the note) from features (the landmark coordinates)
y = df['label'].values
X = df.drop('label', axis=1).values

# Features are already normalized 0-1 by MediaPipe, just make sure dtype is right
X = X.astype('float32')

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode the labels for multi-class classification
num_classes = 7  # do, re, mi, fa, so, la, ti
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print("Data preprocessed successfully.")

# --- 2. Build the Model ---
print("Building the model...")

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.summary()

# --- 3. Train the Model ---
print("\nStarting model training...")

# Build the input pipeline manually with autotune disabled. Keras normally
# wraps numpy arrays in a tf.data.Dataset automatically, and that pipeline's
# background autotuning threads can deadlock with Accelerate BLAS on macOS.
# Disabling autotune and capping the thread pool avoids that.
data_options = tf.data.Options()
data_options.autotune.enabled = False
data_options.experimental_threading.private_threadpool_size = 1
data_options.experimental_threading.max_intra_op_parallelism = 1

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .batch(32) \
    .with_options(data_options)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .batch(32) \
    .with_options(data_options)

history = model.fit(train_ds,
                    epochs=80,
                    validation_data=val_ds,
                    verbose=1)

print("Model training completed.")

# --- 4. Evaluate and Save the Model ---
loss, accuracy = model.evaluate(val_ds)
print(f'\nTest Accuracy: {accuracy*100:.2f}%')

model.save('model/kodaly_model.h5')
print("\nModel saved successfully as model/kodaly_model.h5")