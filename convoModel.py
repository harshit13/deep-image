'''
Image Classification with CIFAR 100
'''

# required imports
import numpy
import os
import sys
from six.moves import cPickle
import matplotlib.pyplot as plt

# keras imports
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as bknd
# from keras.models import load_model


# PREPROCESSING
# parsing dataset. Cifar 100 raw dataset are in bytes
# we need to convert those into (image, label) format
def load_batch(fpath, label_key='labels'):
    '''function parsing CIFAR 100 data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(img, labels)`.
    '''
    # open file to parse
    f = open(fpath, 'rb')

    # check python version here Python 3 is favorable
    if sys.version_info < (3,):
        # Python 3 cPickle directly parses with load()
        # so no extra steps required here
        d = cPickle.load(f)
    else:
        # Python 2 requires extra step to decode the byte information
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded

    # close the file on successfull decoding
    f.close()

    # get (data, labels) from parsed file
    img = d['data']
    labels = d[label_key]

    # reshape data as per required by our network
    # here in cifar 100 we have 32x32 image size
    # with 3 layers (r, g, b)
    img = img.reshape(img.shape[0], 3, 32, 32)

    # return all (img, label) pairs
    return img, labels


# Load train and test data
def load_data(label_mode='fine'):
    '''function load train and test data
    # Arguments
        label_mode: cifar-100 has two label model
            1. fine : 100 classes
            2. coarse : 20 classes
    # Returns
        (x_train, y_train), (x_test, y_test)
    '''

    # folder name, tar name
    dirname = 'cifar-100-python'
    origin = 'cifar-100-python.tar.gz'

    '''get_file function availabe with keras
    # extracts the file if it is compressed
    # else it fetches file pointer to the directory loacation
    '''
    path = get_file(dirname, origin=origin, untar=True)

    # getting training dataset 50000
    fpath = os.path.join(path, 'train')
    X_train, y_train = load_batch(
        fpath, label_key=label_mode + '_labels')

    # getting test dataset 10000
    fpath = os.path.join(path, 'test')
    X_test, y_test = load_batch(
        fpath, label_key=label_mode + '_labels')

    # generates list of labels for test and train set
    y_train = numpy.reshape(y_train, (len(y_train), 1))
    y_test = numpy.reshape(y_test, (len(y_test), 1))

    # sets image data format to 'channel last'
    # as we have image in format 'channel first'
    if bknd.image_data_format() == 'channels_last':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    # set image data format
    bknd.set_image_dim_ordering('th')

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # load datasets
    (X_train, y_train), (X_test, y_test) = load_data('coarse')

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # Create the model
    model = Sequential()

    # feature selection layers
    model.add(Convolution2D(
        32, (3, 3), input_shape=(3, 32, 32), activation='relu',
        padding='same')
    )
    model.add(Dropout(0.2))
    model.add(Convolution2D(
        32, (3, 3), activation='relu', padding='same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(
        64, (3, 3), activation='relu', padding='same'
        )
    )
    model.add(Dropout(0.2))
    model.add(Convolution2D(
        64, (3, 3), activation='relu', padding='same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(
        128, (3, 3), activation='relu', padding='same'
        )
    )
    model.add(Dropout(0.2))
    model.add(Convolution2D(
        128, (3, 3), activation='relu', padding='same'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # classification layers
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(
        1024, activation='relu', kernel_constraint=maxnorm(3)
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(
        512, activation='relu', kernel_constraint=maxnorm(3)
        )
    )
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    epochs = 40
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd, metrics=['accuracy']
    )
    print(model.summary())

    # Fit the model
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        nb_epoch=epochs, batch_size=32
    )

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # save model
    model.save('my_model.h5')
    model.save_weights('my_model_weights.h5')
