import pandas as pd
import numpy as np

#Defining constants here
C = [100/873, 500/873, 700/873]
T = 100
INITIAL_LR = 0.1
LR_DECAY = 0.01


bank_note_train = "bank-note/train.csv" 
bank_note_test = "bank-note/test.csv"

def read_csv(filepath):
    data = pd.read_csv(filepath, header=None)

    # convert binary labels from {0, 1} to {-1, 1}
    data[4] = data[4].apply(lambda label: -1 if label == 0 else 1)

    num_rows = data.values.shape[0]
    num_cols = data.values.shape[1]

    # features and labels arrays
    features = []
    labels = []

    for i in range(num_rows):
        row_data = data.iloc[i].tolist()
        features.append(row_data[0:num_cols-1])
        labels.append(row_data[num_cols-1])

    return np.array(features), np.array(labels)
