## Overview

This project focuses on recognizing handwritten digits using a deep learning model trained on the MNIST dataset. The model is built using TensorFlow and employs a neural network to classify digits (0-9) with high accuracy.

## Tools & Technologies

**Programming Language**: Python

**Libraries**: TensorFlow, NumPy, OpenCV, Matplotlib

**Dataset**: MNIST (Modified National Institute of Standards and Technology database)

**Deep Learning**: Feedforward Neural Network

## Data

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels in size. The dataset is split into:

**Training set**: 60,000 images

**Test set**: 10,000 images

## Model Architecture

The neural network consists of the following layers:

**Flatten Layer**: Converts the 28x28 image into a 1D array.

**Hidden Layers**: Four fully connected (Dense) layers with 128 neurons each and ReLU activation.

**Output Layer**: A fully connected layer with 10 neurons (one per digit) using Softmax activation for classification.

## Process

**Load Dataset**: The MNIST dataset is loaded and split into training and testing sets.

**Data Normalization**: Pixel values are normalized to range [0,1] to improve model performance.

**Model Training**: The model is compiled using Adam optimizer and Sparse Categorical Crossentropy loss function.
It is trained for 2 epochs.

**Evaluation**: Model accuracy and loss are evaluated on the test set.
Results are displayed in a table format.

**Model Saving & Loading**: The trained model is saved as handwritten_digits.model and can be reloaded for future predictions.
