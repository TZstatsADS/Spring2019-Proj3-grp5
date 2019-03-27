#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:11:53 2019

@author: elena

Contains ALL COMPUTATIONAL STEPS

"""
#### LIBRARIES

# Utils

start_1 = time.time()

import os
import glob
import numpy as np
from numpy import array
import random
import math
import time

# Images resizing

import cv2

# CNN

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU
from tensorflow.python.keras.layers import add
from tensorflow.python.keras.optimizers import Adam

# Visualization

import matplotlib.pyplot as plt

end_1 = time.time()

#### STEP 0: DIRECTORIES SPECIFIED

random.seed(2019)
os.getcwd()
path = "D:\\training_images\\"
os.chdir(path)

#### STEP 1: CONTROLS FOR EVALUATION EXPERIMENTS

RUN_CV = True
K_FOLDS = 5
RUN_FEATURE_TRAIN = True
RUN_TEST = True
RUN_FEATURE_TEST = True

#### STEP 2: IMPORTING LABELS (omitted, we don't use labels)

#### STEP 3: FEATURES AND IMAGE TRANSFORMATIONS 
start_2 = time.time()
 
hr_paths = get_paths("HR")
HR_cropped = cropped_imgs(hr_paths)
x_train_lr, x_train_hr, x_test_lr, x_test_hr = train_test_data(HR_cropped, number_of_images = 900)

end_2 = time.time()
#ex = x_train_hr[0]
#### STEP 4a: TRAIN MODEL
start_3 = time.time()

train(epochs = 40, batch_size = 10)

end_3 = time.time()
#### STEP 4b: CROSS-VALIDATION AND PARAMETER TUNING (performance results visualized, best model chosed)

plt.imshow(x_train_lr[0], interpolation='nearest')
plt.imshow(x_train_hr[0], interpolation='nearest')

#### STEP 5: TEST ON HOLDOUT SET

start_4 = time.time()
end_4 = time.time()

#### SUMMARISE RUNNING TIME

lib_download_time = end_1 - start_1
data_prep_time = end_2 - start_2
train_time = end_3 - start_3
test_time = end_4 - start_4
