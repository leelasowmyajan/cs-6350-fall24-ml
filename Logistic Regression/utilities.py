import pandas as pd
import numpy as np

bank_note_train = "bank-note/train.csv" 
bank_note_test = "bank-note/test.csv"

#Defining constants here
P_VARIANCES = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
T = 100
INITIAL_LEARNING_RATE = 0.01
DECAY_RATE = 0.1

def read_csv(filepath):
    data = pd.read_csv(filepath, header=None).values
    
    # convert binary labels from {0, 1} to {-1, 1}
    data[:, -1] = np.where(data[:, -1] == 0, -1, 1)
    
    # add a bias term (column of 1s) to the features
    features = np.hstack((data[:, :-1], np.ones((data.shape[0], 1))))
    
    # labels are the last column
    labels = data[:, -1]
    
    return features, labels

def generate_shuffled_indices(data, T):
    """
    Generates shuffled indices for each epoch of training.
    """
    shuffled_indices = []
    num_samples = data.shape[0]
    for _ in range(T):
        indices = np.arange(num_samples)  # generating sequential indices
        np.random.default_rng().shuffle(indices)  # shuffling them randomly
        shuffled_indices.append(indices)
    return shuffled_indices