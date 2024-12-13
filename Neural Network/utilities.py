import pandas as pd
import numpy as np
import torch
import neural_network

bank_note_train = "bank-note/train.csv" 
bank_note_test = "bank-note/test.csv"

#Defining constants here
######
HIDDEN_LAYER_SIZES = [5, 10, 25, 50, 100]
NETWORK_DEPTHS = [3,5,9]
T = 10
TRAINING_MODES = [0,1]
TRAINING_BATCH_SIZE = 10
######

COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NN_LAYERS = [neural_network.SigmoidAct(), 
          neural_network.SigmoidAct(), 
          neural_network.LinearAct()]

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
        row_data = data.iloc[i].to_numpy()
        feature_vector = np.hstack(([1], row_data[:num_cols-1]))  # prepending 1 to feature vector
        features[i] = feature_vector
        labels[i] = row_data[num_cols-1]

    return features, labels

def create_data_loader(file_path, batch_size, shuffle=False):
    """Creates a DataLoader from a CSV file."""
    data = pd.read_csv(file_path, header=None)

    num_rows = data.values.shape[0]
    num_columns = data.values.shape[1]

    features = []
    labels = []
    for i in range(num_rows):
        row = data.iloc[i].tolist()
        row = list(map(lambda value: np.float32(value), row))
        features.append(row[:num_columns-1])
        labels.append(row[num_columns-1])
    features = np.array(features)
    labels = np.array(labels)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32)
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
