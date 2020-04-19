"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
"""
from tensorflow import keras
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam

def create_CMATER_model(IMG_SIZE=64, N_CHANNELS=1):
    inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Flatten()(model)
    model = Dense(1024, activation = "relu")(model)
    model = Dropout(rate=0.3)(model)
    dense = Dense(512, activation = "relu")(model)

    component = Dense(199, activation = 'softmax')(dense)

    model = Model(inputs=inputs, outputs=component)

    return model