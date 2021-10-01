"""
Copyright: tensorflow.org
Available @ https://www.tensorflow.org/tutorials/images/classification
Recreation in tensorflow 2.5: Amir Hossini
 -
"""
## Libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
from shutil import copyfile
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

## GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## Folder setup
data_folder=f"{os.getcwd()}/Datasets/flower_photos"

## Functions
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
  src_list = os.listdir(SOURCE)
  src_shuf = random.sample(src_list, len(src_list))
  src_clnn = []

  for ifile in src_shuf:
    fpath = os.path.join(SOURCE, ifile)
    fsize = os.path.getsize(fpath)
    if fsize > 0:
      src_clnn.append(ifile)

  src_train = src_clnn[:int(len(src_clnn) * SPLIT_SIZE)]
  src_test = src_clnn[int(len(src_clnn) * SPLIT_SIZE):]

  for itrain in src_train:
    fsrc = os.path.join(SOURCE, itrain)
    fdes = os.path.join(TRAINING, itrain)
    copyfile(fsrc, fdes)

  for itest in src_test:
    fsrc = os.path.join(SOURCE, itest)
    fdes = os.path.join(TESTING, itest)
    copyfile(fsrc, fdes)
  return

def model_fit(num_classes, train_ds,val_ds,epochs=10, callbacks=None):
    model = Sequential([
        layers.InputLayer(input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    print(model.summary())

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs,
    #     callbacks=callbacks
    # )
    return

## Load Data

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
shutil.copytree(data_dir,data_folder,dirs_exist_ok=True)

print('data directory: ',format(data_dir))

"""
Folder structure
flower_photo/
  daisy/
  dandelion/
  roses/
  sunflowers/
  tulips/
"""

## Image count and vizi

image_count = sum([len(files) for r, d, files in os.walk(data_folder)])
print('image count = ', image_count)

roses = list(os.listdir(os.path.join(data_folder,'roses')))
tulips = list(os.listdir(os.path.join(data_folder,'tulips')))
PIL.Image.open(os.path.join(os.path.join(data_folder,'roses'),str(roses[0])))#.show()
PIL.Image.open(os.path.join(os.path.join(data_folder,'tulips'),str(tulips[0])))#.show()

## Train-test split
split_size = 0.8
batch_size = 32
img_height = 180
img_width = 180

shutil.rmtree(f"{os.getcwd()}/tmp", ignore_errors=True)
train_dir      = os.path.join(f"{os.getcwd()}/tmp",'train')
validation_dir = os.path.join(f"{os.getcwd()}/tmp",'validation')

categories    = [dirs for r,dirs,f in os.walk(data_folder)][0]
for catg in categories:
  source_cat    = os.path.join(data_folder,catg)
  train_dir_cat = os.path.join(train_dir,catg)
  os.makedirs(train_dir_cat, exist_ok=True)
  valid_dir_cat = os.path.join(validation_dir,catg)
  os.makedirs(valid_dir_cat, exist_ok=True)
  split_data(source_cat,train_dir_cat,valid_dir_cat,split_size)

nfl_train = sum([len(f) for r,d,f in os.walk(train_dir)])
nfl_valid = sum([len(f) for r,d,f in os.walk(validation_dir)])

dgen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
dgen_valid = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

train_gen = dgen_train.flow_from_directory(train_dir,
                                           batch_size=batch_size,
                                           class_mode='categorical',
                                           target_size=(img_height,img_width)
                                           )
valid_gen = dgen_valid.flow_from_directory(validation_dir,
                                           batch_size=batch_size,
                                           class_mode='categorical',
                                           target_size=(img_height,img_width)
                                           )

history = model_fit(len(categories),train_gen,valid_gen,epochs=10)
