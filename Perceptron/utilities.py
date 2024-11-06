import pandas as pd
import numpy as np

#Defining constants here
T = 10
LEARNING_RATE = 0.1

bank_note_train = "bank-note/train.csv" 
bank_note_test = "bank-note/test.csv"

def read_csv(filepath):
    data = pd.read_csv(filepath, header=None)

    # convert binary labels from {0, 1} to {-1, 1}
    data[4] = data[4].apply(lambda label: -1 if label == 0 else 1)

    num_rows = data.values.shape[0]
    num_cols = data.values.shape[1]
    
    # features and labels arrays
    features = np.empty((num_rows, num_cols))
    labels = np.empty((num_rows, 1))

    for i in range(num_rows):
        row_data = data.iloc[i].tolist()
        features[i] = np.array(row_data[0:num_cols])
        features[i,num_cols-1] = 1 # setting bias term in the last column of features
        labels[i] = row_data[num_cols-1]

    return features, labels