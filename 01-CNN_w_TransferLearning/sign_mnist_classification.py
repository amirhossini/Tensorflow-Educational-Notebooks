"""
Original Version from coursera.org
Available as "C2_W4_Assignment.ipynb" under Course 2 of Tensorflow Professional Certificate by Laurence Moroney

Recreation in Tensorflow 2.5 (& Python 3.8) by Amir Hossini:
 - data source:
        # sign_mnist_train.csv
        wget https://drive.google.com/uc?id=1z0DkA9BytlLxO1C0BAWzknLyQmZAp0HR -O sign_mnist_train.csv
        # sign_mnist_test.csv
        wget https://drive.google.com/uc?id=1z1BIj4qmri59GWBG4ivMNFtpZ4AXIbzg -O sign_mnist_test.csv
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
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import pathlib

from tensorflow.keras.applications.inception_v3 import InceptionV3

## GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(os.getcwd())
