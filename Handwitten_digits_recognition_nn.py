import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_new = True

if train_new:

    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=2)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    #print evaluation
    from prettytable import PrettyTable
    t = PrettyTable(['Parameter', 'Value'])
    t.add_row(['Loss', val_loss])
    t.add_row(['Accuracy', val_acc])
    print(t)

    # Saving the model
    model.save('./handwritten_digits.model')

# Load the model
model = tf.keras.models.load_model('./handwritten_digits.model')

# GUI

from tkinter import *
from PIL import Image, ImageTk, ImageGrab
from tkinter import ttk

window = Tk()

def get_x_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_event(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), fill = 'black', width=50, capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lasx, lasy = event.x, event.y
    
canvas=Canvas(window, width=600, height=500, bg = 'white')
canvas.place(x=120, y=50)
#canvas.pack(anchor='nw') #, fill = 'both', expand = 1)

b1 = ttk.Button(window, text="Quit", command=window.destroy).grid(column=1, row=0)
#b1.place(x=320, y=370)

canvas.bind('<Button-1>', get_x_y)
canvas.bind('<B1-Motion>', draw_event)

# Clears the canvas
def clear_widget():
    canvas.delete("all")

clear_btn = ttk.Button(window, text = "Clear", command = clear_widget).grid(column=1, row=1)
#clear_btn.place(x=320, y=470)

def getPicture():
    #cut part of picture border to avoid having a gray border
    cutoff = 5

    x=window.winfo_rootx()+canvas.winfo_x()
    y=window.winfo_rooty()+canvas.winfo_y()
    x1=x+canvas.winfo_width()
    y1=y+canvas.winfo_height()
    img = ImageGrab.grab().crop((x+cutoff,y+cutoff,x1-cutoff,y1-cutoff)).save("./digit_full.png")
    img = ImageGrab.grab().crop((x+cutoff,y+cutoff,x1-cutoff,y1-cutoff)).resize((28, 28)).save("./digit.png")
    
    # Load custom images and predict them
    img_original = cv2.imread('./digit.png')[:,:,0]
    img_original = np.invert(np.array([img_original]))
    
    # Normalize
    img = img_original/255
    
    prediction = model.predict(img)
    print("The number is probably a {}".format(np.argmax(prediction)))
    #plt.imshow(img[0], cmap=plt.cm.binary)
    #plt.show()
    print(prediction)

    # Displaying the result
    l1 = Label(window, text="The number is probably a {}".format(np.argmax(prediction)), font=('Arial', 15))
    l1.place(x=200, y=650)


classify_btn = ttk.Button(window, text = "Classify", command = getPicture).grid(column=1, row=2)

# Set the size of the tkinter window
window.geometry("750x750")

window.mainloop()