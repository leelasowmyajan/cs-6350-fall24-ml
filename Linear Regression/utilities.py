import numpy as np
import pandas as pd

#defining paths used in Linear Regression folder
concrete_train = "concrete/train.csv"
concrete_test = "concrete/test.csv"

results_path = "results/" 
results_bgd_csv = results_path + "bgd_results.csv"
results_sgd_csv = results_path + "sgd_results.csv"

random_seed_value = 16

def read_csv(filepath):
    """
    Reads a CSV file and splits the data into features and target values
    """
    data = pd.read_csv(filepath, header=None)
    
    num_rows = data.shape[0]
    num_columns = data.shape[1]
    
    # setting feature matrix and target vector 
    feature_matrix = np.empty((num_rows, num_columns - 1))
    target_vector = np.empty((num_rows, 1))

    # go through each row and separate features from the target column
    for i in range(num_rows):
        row_data = data.iloc[i].tolist()  
        feature_matrix[i] = np.array(row_data[0:num_columns - 1])  
        target_vector[i] = row_data[num_columns - 1]  
    
    return np.asmatrix(feature_matrix), np.asmatrix(target_vector)
