# Avengers Face Recognition System

A deep learning-based face recognition system that identifies Avengers characters (Iron Man, Captain America, Thor, Hulk, Black Widow) from face images using transfer learning.

## Overview

This project builds an image classification pipeline that extracts facial features and predicts the correct Avengers character with a confidence score. It uses transfer learning on a pretrained CNN (MobileNetV2) fine-tuned on a custom, cropped-face Avengers dataset.

## Dataset

- 5 classes: Iron Man, Captain America, Thor, Hulk, Black Widow
- 274 cropped face images total (~50-66 images per class)
- Images vary in resolution and were resized to 224x224 for training

Note: This is a small dataset intended for demonstrating the ML pipeline. A production-grade system would require several hundred images per class to generalize across lighting, angle, and occlusion variations.

## Project Pipeline

1. Dataset loading and exploration
2. Data cleaning (corrupt/invalid image removal)
3. Stratified train / validation / test split
4. Preprocessing (resizing, normalization) and data augmentation (rotation, zoom, flip, brightness)
5. Transfer learning model built on MobileNetV2 (also supports ResNet50, EfficientNetB0, VGG16)
6. Two-phase training: feature extraction (frozen base) followed by fine-tuning
7. Regularization via dropout, batch normalization, and class weighting for imbalance
8. Evaluation: accuracy/loss curves, confusion matrix, classification report
9. Prediction system: upload an image and get the predicted character with confidence score

## Model Architecture

MobileNetV2 (ImageNet weights, frozen initially) -> GlobalAveragePooling2D -> BatchNormalization -> Dense(256, ReLU) -> Dropout -> BatchNormalization -> Dense(128, ReLU) -> Dropout -> Dense(5, Softmax)

## Training Details

- Optimizer: Adam (1e-3 for feature extraction, 1e-5 for fine-tuning)
- Loss: Categorical Crossentropy
- Callbacks: Early Stopping, ReduceLROnPlateau, ModelCheckpoint
- Class imbalance handled via computed class weights

## Results

| Metric | Value |
|---|---|
| Test Accuracy | XX.XX% |
| Test Loss | X.XXXX |

Replace the above with your actual run results, and include your confusion matrix and accuracy/loss plots here as images.

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- Matplotlib / Seaborn
- Google Colab

## How to Run

1. Open the notebook in Google Colab
2. Mount Google Drive and upload the dataset zip
3. Run all cells in order (dataset loading through prediction system)
4. Use the prediction cell to upload a new face image and view the predicted character and confidence

## Project Structure
