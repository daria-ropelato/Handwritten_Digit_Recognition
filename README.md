## 📓 Overview

This project focuses on recognizing handwritten digits using a deep learning model trained on the MNIST dataset. The model is built using TensorFlow and Keras and employs a neural network to classify digits (0-9) with high accuracy. Furthermore, it includes a GUI built with Tkinter to allow users to draw digits and get real-time predictions:

**Steps**
- Load and normalize the MNIST dataset.
- Define and train a neural network model.
- Save the trained model.
- Implement a GUI using Tkinter for digit input.
- Capture user-drawn digits, preprocess them, and make predictions.


## Tools & Technologies

**Programming Language**: Python

**Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Tkinter, PIL (Pillow)

**Dataset**: MNIST (Modified National Institute of Standards and Technology database)

**Deep Learning**: Feedforward Neural Network

## Project Structure

**Data Preprocessing**:

- Load MNIST dataset.
- Normalize pixel values.

**Model Architecture**:

- Sequential model with multiple dense layers.
- Uses ReLU activation for hidden layers and Softmax for output.

**Training & Evaluation**:

- Compiles and trains the model.
- Evaluates accuracy on test data.

**GUI for User Interaction**:

- Users can draw digits on a Tkinter canvas.
- The drawn image is processed and classified using the trained model.



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

## Code Snippets

### Model Training
```
import tensorflow as tf
import numpy as np

# Load and preprocess data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2)

# Save the model
model.save('./handwritten_digits.model')
```
### GUI for Digit Recognition
```
from tkinter import *
from PIL import Image, ImageGrab
import numpy as np
import cv2
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('./handwritten_digits.model')

# Initialize GUI
window = Tk()
canvas = Canvas(window, width=600, height=500, bg='white')
canvas.pack()

# Capture image and classify
def classify_digit():
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
    img = np.invert(np.array(img.convert('L')))
    img = img / 255.0
    prediction = model.predict(img.reshape(1, 28, 28))
    print(f"Predicted digit: {np.argmax(prediction)}")

btn = Button(window, text='Classify', command=classify_digit)
btn.pack()
window.mainloop()
```
## Results

- Achieved an accuracy of approximately 98% on test data.
- GUI allows real-time handwritten digit recognition.

## Future Improvements

- Enhance GUI with better preprocessing.
- Implement more robust noise filtering.
- Experiment with Convolutional Neural Networks (CNNs) for better accuracy.

## Demo
TBD
