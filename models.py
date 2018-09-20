"""
The file containing implementations to all of the neural network models used in our experiments. These include a LeNet
model for MNIST, a VGG model for CIFAR and a multilayer perceptron model for dicriminative active learning, among others.
"""

import numpy as np

from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical, multi_gpu_model

class DiscriminativeEarlyStopping(Callback):
    """
    A custom callback for discriminative active learning, to stop the training a little bit before the classifier is
    able to get 100% accuracy on the training set. This makes sure examples which are similar to ones already in the
    labeled set won't have a very high confidence.
    """

    def __init__(self, monitor='acc', threshold=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.improved = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current > self.threshold:
            if self.verbose > 0:
                print("Epoch {e}: early stopping at accuracy {a}".format(e=epoch, a=current))
            self.model.stop_training = True


class DelayedModelCheckpoint(Callback):
    """
    A custom callback for saving the model each time the validation accuracy improves. The custom part is that we save
    the model when the accuracy stays the same as well, and also that we start saving only after a certain amoung of
    iterations to save time.
    """

    def __init__(self, filepath, monitor='val_acc', delay=50, verbose=0, weights=False):

        super(DelayedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.delay = delay
        if self.monitor == 'val_acc':
            self.best = -np.Inf
        else:
            self.best = np.Inf
        self.weights = weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.monitor == 'val_acc':
            current = logs.get(self.monitor)
            if current >= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)
        else:
            current = logs.get(self.monitor)
            if current <= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)



class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus, cpu_relocation=False, cpu_merge=False)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def get_discriminative_model(input_shape):
    """
    The MLP model for discriminative active learning, without any regularization techniques.
    """

    if np.sum(input_shape) < 30:
        width = 20
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(2, activation='softmax', name='softmax'))
    else:
        width=256
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(2, activation='softmax', name='softmax'))

    return model


def get_LeNet_model(input_shape, labels=10):
    """
    A LeNet model for MNIST.
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='embedding'))
    model.add(Dropout(0.5))
    model.add(Dense(labels, activation='softmax', name='softmax'))

    return model


def get_VGG_model(input_shape, labels=10):
    """
    A VGG model for CIFAR.
    """

    weight_decay = 0.0005
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay), name='embedding'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(labels, activation='softmax', name='softmax'))

    return model


def get_autoencoder_model(input_shape, labels=10):
    """
    An autoencoder for MNIST to be used in the DAL implementation.
    """

    image = Input(shape=input_shape)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(image)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = UpSampling2D((2, 2), name='embedding')(encoder)
    decoder = Conv2D(4, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)

    autoencoder = Model(image, decoder)
    return autoencoder


def train_discriminative_model(labeled, unlabeled, input_shape, gpu=1):
    """
    A function that trains and returns a discriminative model on the labeled and unlabaled data.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0],1),dtype='int')
    y_U = np.ones((unlabeled.shape[0],1),dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    Y_train = to_categorical(Y_train)

    # build the model:
    model = get_discriminative_model(input_shape)

    # train the model:
    batch_size = 1024
    if np.max(input_shape) == 28:
        optimizer = optimizers.Adam(lr=0.0003)
        epochs = 200
    elif np.max(input_shape) == 128:
        # optimizer = optimizers.Adam(lr=0.0003)
        # epochs = 200
        batch_size = 128
        optimizer = optimizers.Adam(lr=0.0001)
        epochs = 1000 #TODO: was 200
    elif np.max(input_shape) == 512:
        optimizer = optimizers.Adam(lr=0.0002)
        # optimizer = optimizers.RMSprop()
        epochs = 500
    elif np.max(input_shape) == 32:
        optimizer = optimizers.Adam(lr=0.0003)
        epochs = 500
    else:
        optimizer = optimizers.Adam()
        # optimizer = optimizers.RMSprop()
        epochs = 1000
        batch_size = 32

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DiscriminativeEarlyStopping()]
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks,
              class_weight={0 : float(X_train.shape[0]) / Y_train[Y_train==0].shape[0],
                            1 : float(X_train.shape[0]) / Y_train[Y_train==1].shape[0]},
              verbose=2)

    return model


def train_mnist_model(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    """
    A function that trains and returns a LeNet model on the labeled MNIST data.
    """

    if K.image_data_format() == 'channels_last':
        input_shape = (28, 28, 1)
    else:
        input_shape = (1, 28, 28)

    model = get_LeNet_model(input_shape=input_shape, labels=10)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(filepath=checkpoint_path, verbose=1, weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus = gpu)
        gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        gpu_model.fit(X_train, Y_train,
                      epochs=150,
                      batch_size=32,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        del model
        del gpu_model

        model = get_LeNet_model(input_shape=input_shape, labels=10)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.load_weights(checkpoint_path)
        return model

    else:
        model.fit(X_train, Y_train,
                  epochs=150,
                  batch_size=32,
                  shuffle=True,
                  validation_data=(X_validation, Y_validation),
                  callbacks=callbacks,
                  verbose=2)
        model.load_weights(checkpoint_path)
        return model


def train_cifar10_model(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    """
    A function that trains and returns a VGG model on the labeled CIFAR-10 data.
    """

    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)

    model = get_VGG_model(input_shape=input_shape, labels=10)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(filepath=checkpoint_path, verbose=1, weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus = gpu)
        gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        gpu_model.fit(X_train, Y_train,
                      epochs=400,
                      batch_size=32,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        del gpu_model
        del model

        model = get_VGG_model(input_shape=input_shape, labels=10)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.load_weights(checkpoint_path)

        return model

    else:
        model.fit(X_train, Y_train,
                      epochs=400,
                      batch_size=32,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        model.load_weights(checkpoint_path)
        return model


def train_cifar100_model(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    """
    A function that trains and returns a VGG model on the labeled CIFAR-100 data.
    """

    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)

    model = get_VGG_model(input_shape=input_shape, labels=100)
    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(filepath=checkpoint_path, verbose=1, weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus = gpu)
        gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        gpu_model.fit(X_train, Y_train,
                      epochs=1000,
                      batch_size=128,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        del gpu_model
        del model

        model = get_VGG_model(input_shape=input_shape, labels=100)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.load_weights(checkpoint_path)

        return model

    else:
        model.fit(X_train, Y_train,
                      epochs=1000,
                      batch_size=128,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        model.load_weights(checkpoint_path)
        return model


