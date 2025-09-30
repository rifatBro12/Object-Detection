#!/usr/bin/env python3
"""
Lightweight CIFAR-10 training script optimized for 6GB GPU
Uses a smaller CNN architecture instead of ResNet50
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, models, layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def configure_gpu():
    """Configure GPU memory growth"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} GPU(s)")
            print(f"GPU name: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.test.is_gpu_available()}")

def load_and_preprocess_data():
    """Load and preprocess CIFAR-10 data"""
    print("Loading CIFAR-10 data...")
    
    # Load labels
    labels_df = pd.read_csv(r"C:\Users\rifat\deep learning projects\cifar-10\trainLabels.csv")
    
    # Create labels dictionary
    labels_dictionary = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    labels = [labels_dictionary[i] for i in labels_df['label']]
    
    # Get image IDs
    id_list = list(labels_df['id'])
    
    # Load images (use a subset for faster training)
    train_data_folder = 'C:/Users/rifat/deep learning projects/cifar-10/train/train/'
    data = []
    
    print("Loading images (using subset for memory efficiency)...")
    # Use only first 10000 images for faster training and less memory usage
    subset_size = 10000
    for i, id in enumerate(id_list[:subset_size]):
        if i % 1000 == 0:
            print(f"Loaded {i}/{subset_size} images...")
        image = Image.open(train_data_folder + str(id) + '.png')
        data.append(np.array(image))
    
    # Convert to numpy arrays
    X = np.array(data)
    Y = np.array(labels[:subset_size])
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {Y.shape}")
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    # Scale data
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, Y_train, Y_test

def create_lightweight_model():
    """Create a lightweight CNN model optimized for 6GB GPU"""
    print("Creating lightweight CNN model...")
    
    num_of_classes = 10
    
    # Create a simpler CNN model suitable for 32x32 images
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')
    ])
    
    print("Lightweight CNN model created")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

def train_model(model, X_train_scaled, Y_train, X_test_scaled, Y_test):
    """Train the model with GPU optimization"""
    print("Compiling model...")
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Use smaller batch size for 6GB GPU memory
    batch_size = 32  # Can use larger batch size with lightweight model
    
    # Add callbacks for model saving and early stopping
    callbacks = [
        ModelCheckpoint(
            'cifar10_lightweight_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"Starting training with batch size: {batch_size}")
    print("Using lightweight CNN architecture for 6GB GPU")
    
    # Train the model
    history = model.fit(
        X_train_scaled, 
        Y_train, 
        validation_split=0.1, 
        epochs=50,  # More epochs since model is smaller
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, batch_size

def evaluate_and_save_model(model, X_test_scaled, Y_test, batch_size):
    """Evaluate model and save it"""
    print("Evaluating model...")
    
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test_scaled, Y_test, batch_size=batch_size, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the final model
    model.save('cifar10_lightweight_final_model.h5')
    print("Model saved as 'cifar10_lightweight_final_model.h5'")
    
    # Save model architecture and weights separately
    model_json = model.to_json()
    with open("cifar10_lightweight_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("cifar10_lightweight_weights.weights.h5")
    print("Model architecture saved as 'cifar10_lightweight_model.json'")
    print("Model weights saved as 'cifar10_lightweight_weights.weights.h5'")
    
    # Create and save class labels
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    with open('class_labels.json', 'w') as f:
        json.dump(class_labels, f)
    print("Class labels saved to 'class_labels.json'")
    
    # Test prediction on a sample image
    sample_image = X_test_scaled[0:1]
    prediction = model.predict(sample_image, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    print(f"\nSample prediction test:")
    print(f"Predicted class: {class_labels[predicted_class]} (index: {predicted_class})")
    print(f"Confidence: {confidence:.4f}")
    print(f"Actual class: {class_labels[Y_test[0]]}")

def main():
    print("=" * 60)
    print("CIFAR-10 Lightweight Object Detection Training System")
    print("=" * 60)
    
    # Configure GPU
    configure_gpu()
    
    # Load and preprocess data
    X_train_scaled, X_test_scaled, Y_train, Y_test = load_and_preprocess_data()
    
    # Create model
    model = create_lightweight_model()
    
    # Train model
    history, batch_size = train_model(model, X_train_scaled, Y_train, X_test_scaled, Y_test)
    
    # Evaluate and save model
    evaluate_and_save_model(model, X_test_scaled, Y_test, batch_size)
    
    print("\nTraining completed successfully!")
    print("Model files saved:")
    print("- cifar10_lightweight_model.h5 (best model)")
    print("- cifar10_lightweight_final_model.h5 (final model)")
    print("- cifar10_lightweight_model.json (architecture)")
    print("- cifar10_lightweight_weights.h5 (weights)")
    print("- class_labels.json (class labels)")
    print("\nReady to run the prediction app: streamlit run app.py")

if __name__ == "__main__":
    main()
