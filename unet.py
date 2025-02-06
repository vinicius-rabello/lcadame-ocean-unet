import numpy as np
from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
import keras
import keras.backend as K
from keras.callbacks import History
history = History()

import keras
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D
from keras.models import load_model
from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

### Define Data-driven architecture ######
def stn(input_shape=(192, 96,2), sampling_size=(8, 16), num_classes=10):
    inputs = Input(shape=input_shape)
    conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv3)

    conv5 = Convolution2D(32, (5, 5), activation='relu', padding='same')(pool2)
    x = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv5)

    up6 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x)), conv2])
    conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv6)

    up7 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv1])
    conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(32, (5, 5), activation='relu', padding='same')(conv7)

### Use tanh in this last layer
    conv10 = Convolution2D(2, (5, 5), activation='linear',padding='same')(conv7)

#    model = Model(input=inputs, output=conv10)
    model = Model(inputs, conv10)

    return model