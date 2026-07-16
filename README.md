# Avengers Face Recognition System

A deep learning-based face recognition system that identifies five Avengers characters using Transfer Learning with **MobileNetV2**.

## Overview

This project uses a pretrained CNN to classify cropped face images into one of the following characters:

- Iron Man
- Captain America
- Thor
- Hulk
- Black Widow

The model predicts the character along with a confidence score.

## Dataset

- 5 Classes
- 274 cropped face images
- Images resized to **224 × 224**
- Data augmentation applied to improve generalization

## Project Pipeline

- Dataset Loading
- Data Cleaning
- Train/Validation/Test Split
- Image Preprocessing
- Data Augmentation
- Transfer Learning (MobileNetV2)
- Feature Extraction & Fine-Tuning
- Model Evaluation
- Face Prediction System

## Model Architecture

```
MobileNetV2
      ↓
GlobalAveragePooling2D
      ↓
BatchNormalization
      ↓
Dense (256)
      ↓
Dropout
      ↓
Dense (128)
      ↓
Dropout
      ↓
Softmax (5 Classes)
```

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Google Colab

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 79.07% |
| Test Loss | 0.6675 |


## Project Structure

```
marvel-face-recognition-cnn/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
└── models/
    ├── avengers_face_recognition_final.keras
    ├── avengers_face_recognition_final.h5
    └── class_indices.json
```

## How to Run

```bash
git clone https://github.com/shikhar11x/marvel-face-recognition-cnn.git
cd marvel-face-recognition-cnn
pip install -r requirements.txt
```

Run the notebook in **Google Colab**, train the model, and use the prediction cell to classify new face images.

## Author

**Shikhar Bajpai**

GitHub: https://github.com/shikhar11x  
LinkedIn: https://linkedin.com/in/shikharbajpai1
