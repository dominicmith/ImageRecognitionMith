
# Image Recognition 

## Overview

This project aims to build an image recognition system using convolutional neural networks (CNNs) with TensorFlow and Keras. The model is trained on the CIFAR-10 dataset to classify images into 10 different classes.

## Files

- model.h5: This file contains the trained CNN model saved in HDF5 format.
- cifar10_model.py**: Python script used to train the CIFAR-10 model.
- app.py: Flask backend script for serving predictions via REST API.
- frontend: React frontend folder for uploading images and displaying predictions.

## Setup Instructions

### Backend (Flask)

1. Install required Python packages:
   ```
   pip install Flask tensorflow numpy opencv-python flask-cors
   ```

2. Run Flask app:
   ```
   python app.py
   ```

### Frontend (React)

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start React app:
   ```
   npm start
   ```

## Usage

1. Upload an image through the frontend interface.
2. Click on the "Upload & Predict" button to trigger the prediction.
3. View the predicted class label displayed on the screen.

## Model Evaluation

The model's performance metrics (accuracy, precision, recall) were evaluated on the CIFAR-10 test set. The trained model achieved an accuracy of approximately 90% on the test data.

## Future Improvements

- Experiment with different pre-trained models (e.g., ResNet, VGG) to potentially improve accuracy.
- Enhance the frontend interface for better user experience.
- Implement more sophisticated error handling and logging in the Flask backend.

## Credits

- TensorFlow: Used for building and training the CNN model.
- React: Frontend framework for user interface development.
- Flask: Backend framework for serving predictions.

