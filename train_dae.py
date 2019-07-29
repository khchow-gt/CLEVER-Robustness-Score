import os
import numpy as np
import keras.regularizers as regs
from keras.layers.core import Lambda
from setup_mnist import MNIST
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D, UpSampling2D, AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda


class DAE:
    def __init__(self):
        self.model_dir = "./dae/"
        self.v_noise = 0.1
        h, w, c = [28, 28, 1]

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
        model.add(Conv2D(c, (3, 3), activation='sigmoid', padding='same', activity_regularizer=regs.l2(1e-9)))

        model.add(Lambda(lambda x_: x_ - 0.5))

        self.model = model

    def train(self, data, archive_name, num_epochs=100, batch_size=256, if_save=True):
        self.model.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer='adam')

        noise = self.v_noise * np.random.normal(size=np.shape(data.train_data))
        noisy_train_data = data.train_data + noise
        noisy_train_data = np.clip(noisy_train_data, -0.5, 0.5)

        self.model.fit(noisy_train_data, data.train_data,
                       batch_size=batch_size,
                       validation_data=(data.validation_data, data.validation_data),
                       epochs=num_epochs,
                       shuffle=True)

        if if_save:
            self.model.save(os.path.join(self.model_dir, archive_name))

    def load(self, archive_name, model_dir=None):
        if model_dir is None: model_dir = self.model_dir
        self.model.load_weights(os.path.join(model_dir, archive_name))


if __name__ == '__main__':
    AE = DAE()
    AE.train(MNIST(), "mnist")
