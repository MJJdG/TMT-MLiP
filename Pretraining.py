import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2
import os
import datetime

from utils import *
#from view_data import *
from preprocessing import *
from CMATER_model import *

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from keras.callbacks import ReduceLROnPlateau
import albumentations

os.chdir("C:/Users/Ruben/Documents/Master_AI_CNS/MLiP/CMATERdb3.1.3.2/")

# Load Data
x_train = pd.DataFrame(np.load('x_Train.npy')) # Can we load it as pandas?
x_test = pd.DataFrame(np.load('x_Test.npy'))
y_train = pd.DataFrame(np.load('y_Train.npy'))
y_test = pd.DataFrame(np.load('y_Test.npy'))




# Check shape
print(x_train.shape)
print(x_test.shape)


#plt.imshow(x_train[1].reshape(64, 64))
#plt.show()


CMATER_model = create_CMATER_model()
CMATER_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Set a learning rate annealer
learning_rate_reduction_component = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)

batch_size = 64#256
epochs = 30
HEIGHT = 137
WIDTH = 236


x_train = x_train/255
x_test = x_test/255

# CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
x_train = x_train.values.reshape((x_train.shape[0], 64,64, 1))
x_test = x_test.values.reshape((x_test.shape[0], 64,64, 1))


# Convert categorical variable into dummy/indicator variables
y_train = getdummies(y_train)
y_test = getdummies(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Data augmentation for creating more training data
datagen = MultiOutputDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.15, # Randomly zoom image 
    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images


# This will just calculate parameters required to augment the given data. This won't perform any augmentations
datagen.fit(x_train)

# Fit the model

history = CMATER_model.fit_generator(datagen.flow(x_train, {'dense_3': y_train}, batch_size=batch_size),
                            epochs = epochs, validation_data = (x_test, y_test), 
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            callbacks=[learning_rate_reduction_component])

#history = CMATER_model.fit(x=x_train,y=y_train,batch_size=batch_size, epochs= epochs,verbose =1,validation_data=(x_test,y_test))


CMATER_model.save('CMATER_model_30.h5')