"""
This Python file provides a good example of working with image files in text (.csv format).

Original Version from coursera.org (assignments)
Available as "C2_W4_Assignment.ipynb" as assignment 4 for Course 2 of Tensorflow Professional Certificate by Laurence Moroney

- data source:
        # sign_mnist_train.csv
        wget https://drive.google.com/uc?id=1z0DkA9BytlLxO1C0BAWzknLyQmZAp0HR -O sign_mnist_train.csv
        # sign_mnist_test.csv
        wget https://drive.google.com/uc?id=1z1BIj4qmri59GWBG4ivMNFtpZ4AXIbzg -O sign_mnist_test.csv

Recreation in Tensorflow 2.5 (& Python 3.8) by Amir Hossini:
- get_data function
 """
## Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import random
import shutil
from shutil import copyfile
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import pathlib

## GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## Folder setup
data_folder=f"{os.getcwd()}/datasets/sign_mnist"
pret_folder=f"{os.getcwd()}/pretrained_weights"

## Parameters
seed             = 42

split_size       = 0.8
batch_size_train = 32
batch_size_valid = 16

num_classes      = 26

img_height       = 28
img_width        = 28

tl_img_height    = 150 # Adjusted to conform to Inception Input Layer: 150
tl_img_width     = 150 # Adjusted to conform to Inception Input Layer: 150

max_n_epochs     = 20

## Set random seed
tf.random.set_seed=seed

## Functions
class Callback_set(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') is not None and logs.get('val_accuracy') > 0.95):
            print(f"\nReached 95% validation accuracy so cancelling training!")
            self.model.stop_training = True

def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter=',')
        imgs = []
        labels = []

        next(reader, None)

        for row in reader:
            label = row[0]
            data = row[1:]
            img = np.array(data).reshape((28, 28))

            imgs.append(img)
            labels.append(label)

        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
    return images, labels

def model_compile(num_classes,img_height, img_width):
    model = Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation=tf.keras.activations.softmax)
    ])

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

## Load Data
path_sign_mnist_train = f"{os.getcwd()}/datasets/sign_mnist/sign_mnist_train.csv"
path_sign_mnist_test = f"{os.getcwd()}/datasets/sign_mnist/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

print(f'training_images.shape: {training_images.shape}')
print(f'training_labels.shape: {training_labels.shape}')
print(f'testing_images.shape: {testing_images.shape}')
print(f'testing_labels.shape: {testing_labels.shape}')

training_images = np.expand_dims(training_images,axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

print(f'training_images.shape(new): {training_images.shape}')
print(f'training_labels.shape(new): {training_labels.shape}')
print(f'testing_images.shape(new): {testing_images.shape}')
print(f'testing_labels.shape(new): {testing_labels.shape}')

## Image Data Generator
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale = 1./255)

## Train model
train_gen = train_datagen.flow(training_images,training_labels,batch_size=64)
test_gen  = test_datagen.flow(testing_images,testing_labels,batch_size=64)

model   = model_compile(num_classes,img_height,img_width)
history = model.fit(train_gen, validation_data=test_gen,epochs=20)

# Plot the chart for accuracy and loss on both training and validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()