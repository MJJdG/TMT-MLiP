import os
import pandas as pd
import numpy as np
from scipy import misc
from preprocessing import *
import imageio
import cv2
import matplotlib.pyplot as plt


os.chdir("C:/Users/Ruben/Documents/Master_AI_CNS/MLiP/CMATERdb3.1.3.2/FinalDatabase/")


def CMATER_to_npy(train_or_test):
    dataset = []
    i = 0
    for root, _, files in os.walk(f"{train_or_test}/"):
        for name in files:

            # Get Image
            path = os.path.join(root, name)
            image = cv2.imread(path,0)

            # Get Shape
            height, width = image.shape


            if (height > 137):
                image = pd.DataFrame(image)
                image = image.iloc[:136,:]
                height = image.shape[0]

            if (width > 236):
                image = pd.DataFrame(image)
                image = image.iloc[:,:235]
                width = image.shape[1]
                

            if (height <= 137 and width <= 236):

                # Add a tail of 254s to make image fit (137,236)
                # and make sure there's an upper right bound without zeros
                width_tail = np.repeat(254, (height * (236 - width)))
                height_tail = np.repeat(255, ((137 - height) * 236))
                width_tail = width_tail.reshape((height,236-width))
                height_tail = height_tail.reshape((137 - height, 236))

                # Apply tail
                image = np.concatenate((image,width_tail),1)
                image = np.concatenate((image,height_tail),0).astype('uint8')            
                
                # Flatten & Append
                image = image.flatten()
                dataset.append(image)

            else:
                print(height, width, path)


    # Resize dataset
    dataset = pd.DataFrame(np.array(dataset))
    dataset = resize(dataset)
    dataset = np.array(dataset)

    # Save Dataset
    np.save(f"C:/Users/Ruben/Documents/Master_AI_CNS/MLiP/CMATERdb3.1.3.2/x_{train_or_test}",dataset)

def CMATER_to_labels(train_or_test):
    labels = []
    
    label = 0
    for root, dirs, files in os.walk(f"{train_or_test}/", topdown=True):
        num_files = len(files)

        if not num_files is 0:
            new_labels = np.repeat(label, num_files)
            labels = np.concatenate((labels, new_labels))
            label += 1
    
    labels = np.array(labels)
    np.save(f"C:/Users/Ruben/Documents/Master_AI_CNS/MLiP/CMATERdb3.1.3.2/y_{train_or_test}",labels)


#CMATER_to_labels("Test")
#CMATER_to_labels("Train")

CMATER_to_npy("Train")
CMATER_to_npy("Test")

