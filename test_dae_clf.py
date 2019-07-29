import os
import numpy as np
import keras.regularizers as regs
from keras.layers.core import Lambda
from setup_mnist import MNIST
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D, UpSampling2D, AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras.optimizers import SGD
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_dae():
    model = Sequential()

    model.add(Lambda(lambda x_: x_ + 0.5, input_shape=(28, 28, 1)))

    # Encoder
    model.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    model.add(AveragePooling2D((2, 2), padding="same"))
    model.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))

    # Decoder
    model.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9)))

    model.add(Lambda(lambda x_: x_ - 0.5))

    model.load_weights("./dae/mnist")
    model.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='adam')

    return model


def get_clf():
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))

    model.load_weights("./models/mnist")

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])

    return model


def get_dae_clf():
    model1 = Sequential()

    model1.add(Lambda(lambda x_: x_ + 0.5, input_shape=(28, 28, 1)))

    # Encoder
    model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    model1.add(AveragePooling2D((2, 2), padding="same"))
    model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))

    # Decoder
    model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    model1.add(UpSampling2D((2, 2)))
    model1.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same", activity_regularizer=regs.l2(1e-9)))
    model1.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9)))

    model1.add(Lambda(lambda x_: x_ - 0.5))

    model1.load_weights("./dae/mnist")
    model1.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='adam')

    model2 = Sequential()

    model2.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model2.add(Activation('relu'))
    model2.add(Conv2D(32, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Conv2D(64, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(Conv2D(64, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Flatten())
    model2.add(Dense(200))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(200))
    model2.add(Activation('relu'))
    model2.add(Dense(10))

    model2.load_weights("./models/mnist")

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

    model2.compile(loss=fn, optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

    model = Sequential()
    model.add(model1)
    model.add(model2)
    model.compile(loss=fn, optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

    return model


if __name__ == '__main__':
    dae = get_dae()
    clf = get_clf()
    dc = get_dae_clf()

    data = MNIST()
    y_true = np.argmax(data.test_labels, axis=1)

    X_raw = data.test_data
    X_dae = dae.predict(X_raw)

    y_pred_raw = np.argmax(clf.predict(X_raw), axis=1)
    y_pred_dae = clf.predict(X_dae)
    y_pred_dc = dc.predict(X_raw)
    print np.equal(y_pred_dae, y_pred_dc)
    exit()
    print "RAW: %.4f" % np.mean(y_pred_raw == y_true)
    print "DAE: %.4f" % np.mean(y_pred_dae == y_true)
    print "DC : %.4f" % np.mean(y_pred_dc == y_true)
