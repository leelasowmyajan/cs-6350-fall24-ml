import numpy as np
import utilities

def main():
    # loading the training features and targets from the dataset
    features_matrix, target_values = utilities.read_csv(utilities.concrete_train)
    
    # Transpose the feature matrix
    transposed_features = np.transpose(features_matrix)
    
    # calculating the inverse of (X^T * X)
    inverse_term = np.linalg.inv(np.matmul(transposed_features, features_matrix))
    
    # calculating (X^T * y)
    product_term = np.matmul(transposed_features, target_values)
    
    # calculating the optimal weight vector
    optimal_weights = np.matmul(inverse_term, product_term)
    
    print("Analytical weights", optimal_weights)

if __name__ == "__main__":
    main()