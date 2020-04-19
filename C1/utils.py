import pandas as pd
import numpy as np
from keras.callbacks import ReduceLROnPlateau

def read_input_files():
    train_df_ = pd.read_csv('bengaliai-cv19/train.csv')
    test_df_ = pd.read_csv('bengaliai-cv19/test.csv')
    class_map_df = pd.read_csv('bengaliai-cv19/class_map_corrected.csv')
    sample_sub_df = pd.read_csv('bengaliai-cv19/sample_submission.csv')

    return train_df_, test_df_, class_map_df, sample_sub_df

def getdummies(df):
    dummy_df = np.zeros((df.shape[0],199))
    for i in range(len(df)):
        dummy_df[i,int(df.iloc[i])] = 1

    return pd.DataFrame(dummy_df)


def set_learning_rates():
    # Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
    learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_3_accuracy', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)
    learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_4_accuracy', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)
    learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_5_accuracy', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)

    return (learning_rate_reduction_root, learning_rate_reduction_vowel, 
        learning_rate_reduction_consonant)