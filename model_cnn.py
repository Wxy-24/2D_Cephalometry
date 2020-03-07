import tensorflow as tf
import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from keras import backend as K


#define loss function
def euclidean_distance(y_true, y_pred):
  return K.mean(K.sqrt(K.sum(K.square(y_pred-y_true),axis=-1)))

#define model
def yolov1(pretrained_weights=None, input_size=(544, 480, 1)):
    # Convolution block 0
    inputs = Input(input_size)
    conv0 = Conv2D(16, 7, strides=(2, 2), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(inputs)
    BN0 = BatchNormalization()(conv0)
    pool0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(BN0)

    # Convolution block 1
    conv1 = Conv2D(32, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(pool0)
    BN1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(BN1)

    # Convolution block 2
    conv2 = Conv2D(16, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = Conv2D(16, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2 = Conv2D(32, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv2)
    BN2 = BatchNormalization()(conv2)
    merge2 = add([pool1, BN2])
    pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(merge2)
    pool2 = Conv2D(128, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(pool2)

    # Convolution block 3
    conv3 = Conv2D(32, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(32, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(64, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(32, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(64, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(32, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(64, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(64, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(128, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv3)
    BN3 = BatchNormalization()(conv3)
    merge3 = add([pool2, BN3])
    pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(merge3)
    pool3_p = Conv2D(256, 3, strides=(2, 2), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                     kernel_initializer='he_normal')(pool3)

    # Convolution block 4
    conv4 = Conv2D(128, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(256, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(128, 1, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(256, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(256, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(256, 3, strides=(2, 2), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv4)
    BN4 = BatchNormalization()(conv4)
    merge4 = add([pool3_p, BN4])

    # Convolution block 5
    conv5 = Conv2D(256, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(merge4)
    conv5 = Conv2D(256, 3, strides=(1, 1), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same',
                   kernel_initializer='he_normal')(conv5)
    BN5 = BatchNormalization()(conv5)
    merge5 = add([merge4, BN5])

    # flatten and softmax
    dense = Flatten()(merge5)
    dense = Dense(units=256, activation='relu')(dense)
    dense = Dense(units=128, activation='relu')(dense)
    dense = Dense(units=1024, activation='relu')(dense)
    # dropout=Dropout(0.2)(dense)
    coordinate = Dense(units=2, activation='linear')(dense)

    model = Model(inputs=inputs, outputs=coordinate)

    # Define the optimizer
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Compile the model
    model.compile(loss=euclidean_distance, optimizer='Adam', metrics=["accuracy"])

    return model


