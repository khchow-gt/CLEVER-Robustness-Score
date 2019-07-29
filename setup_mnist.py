# setup_mnist.py -- mnist data and model loading code
#
# Copyright (C) 2017-2018, IBM Corp.
# Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
#                and Huan Zhang <ecezhang@ucdavis.edu>
# Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
#
# This program is licenced under the Apache 2.0 licence,
# contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import gzip
import pickle

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.layers import Conv2D, UpSampling2D, AveragePooling2D
import keras.regularizers as regs


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * 28 * 28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255.0) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class MNIST:
    def __init__(self):
        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModel:
    # def __init__(self, restore=None, session=None, use_softmax=False, use_brelu=False, activation="relu"):
    #     def bounded_relu(x):
    #         return K.relu(x, max_value=1)
    #
    #     if use_brelu:
    #         activation = bounded_relu
    #
    #     print("inside MNISTModel: activation = {}".format(activation))
    #
    #     self.num_channels = 1
    #     self.image_size = 28
    #     self.num_labels = 10
    #
    #     model1 = Sequential()
    #
    #     model1.add(Lambda(lambda x_: x_ + 0.5, input_shape=(28, 28, 1)))
    #
    #     # Encoder
    #     model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    #     model1.add(AveragePooling2D((2, 2), padding="same"))
    #     model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    #
    #     # Decoder
    #     model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    #     model1.add(UpSampling2D((2, 2)))
    #     model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    #     model1.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9)))
    #
    #     model1.add(Lambda(lambda x_: x_ - 0.5))
    #
    #     model1.load_weights("./dae/mnist")
    #     model1.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='adam')
    #
    #
    #     model2 = Sequential()
    #
    #     model2.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    #     model2.add(Activation(activation))
    #     model2.add(Conv2D(32, (3, 3)))
    #     model2.add(Activation(activation))
    #     model2.add(MaxPooling2D(pool_size=(2, 2)))
    #
    #     model2.add(Conv2D(64, (3, 3)))
    #     model2.add(Activation(activation))
    #     model2.add(Conv2D(64, (3, 3)))
    #     model2.add(Activation(activation))
    #     model2.add(MaxPooling2D(pool_size=(2, 2)))
    #
    #     model2.add(Flatten())
    #     model2.add(Dense(200))
    #     model2.add(Activation(activation))
    #     model2.add(Dense(200))
    #     model2.add(Activation(activation))
    #     model2.add(Dense(10))
    #     # output log probability, used for black-box attack
    #     if use_softmax:
    #         model2.add(Activation('softmax'))
    #     if restore:
    #         model2.load_weights(restore)
    #
    #     layer_outputs = []
    #     for layer in model1.layers:
    #         if isinstance(layer, Conv2D) or isinstance(layer, Dense):
    #             layer_outputs.append(K.function([model1.layers[0].input], [layer.output]))
    #     for layer in model2.layers:
    #         if isinstance(layer, Conv2D) or isinstance(layer, Dense):
    #             layer_outputs.append(K.function([model2.layers[0].input], [layer.output]))
    #
    #     model = Sequential()
    #     model.add(model1)
    #     model.add(model2)
    #     self.model = model
    #     self.layer_outputs = layer_outputs
    def __init__(self, restore=None, session=None, use_softmax=False, use_brelu=False, activation="relu"):
        def bounded_relu(x):
            return K.relu(x, max_value=1)

        if use_brelu:
            activation = bounded_relu

        print("inside MNISTModel: activation = {}".format(activation))

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(32, (3, 3),
                         input_shape=(28, 28, 1)))
        model.add(Activation(activation))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation(activation))
        model.add(Dense(200))
        model.add(Activation(activation))
        model.add(Dense(10))
        # output log probability, used for black-box attack
        if use_softmax:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        layer_outputs = []
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        self.model = model
        self.layer_outputs = layer_outputs

    def predict(self, data):
        return self.model(data)


class TwoLayerMNISTModel:
    def __init__(self, restore=None, session=None, use_softmax=False):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(1024))
        model.add(Lambda(lambda x: x * 10))
        model.add(Activation('softplus'))
        model.add(Lambda(lambda x: x * 0.1))
        model.add(Dense(10))
        # output log probability, used for black-box attack
        if use_softmax:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        layer_outputs = []
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        self.layer_outputs = layer_outputs
        self.model = model

    def predict(self, data):

        return self.model(data)
