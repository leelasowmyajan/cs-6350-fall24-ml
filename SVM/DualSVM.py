import numpy as np
from scipy.optimize import minimize
import utilities

class DualSVM():
    def __init__(self):
        self.weights = None 
        self.bias = 0  # stores the bias term
        self.alphas = None  # stores the dual variables (alphas)
        self.support_vector = []  # stores indices of support vectors
        self.threshold = 1e-10  # threshold for identifying support vectors

    def train(self, features, labels, C):
        '''
        Trains the svm model using the dual formulation
        '''
        num_samples = features.shape[0]

        initial_alphas = np.zeros(num_samples) 

        # minimizing the dual objective function with constraints and bounds
        result = minimize(
            self.compute_dual_objective, 
            initial_alphas, 
            args=(features, labels), 
            method='SLSQP', 
            constraints=[{'type': 'eq', 'fun': lambda alphas: np.dot(alphas, labels)}], 
            bounds=[(0,C)] * num_samples)
        
        # storing the optimized alpha values
        self.alphas = result.x

        # identifying support vectors based on alpha values greater than the threshold
        self.support_vectors = [i for i, alpha in enumerate(self.alphas) if alpha > self.threshold]

        # initializing weights to zero and calculating them using alpha, labels, and features
        self.weights = np.zeros_like(features[0])
        for i in range(num_samples):
            self.weights += self.alphas[i] * labels[i] * features[i]

        self.bias = 0

        # calculating the bias using support vectors
        for i in self.support_vectors:
            self.bias += labels[i] - self.compute_kernel_single(self.weights, features[i])

        # averaging the bias over all support vectors
        self.bias = self.bias / len(self.support_vectors)

    def compute_linear_kernel(self, feature_set_1, feature_set_2):
        # computing the linear kernel (dot product) between two sets of features
        return feature_set_1 @ feature_set_2.T
    
    def compute_kernel_single(self, vector_1, vector_2):
        # computing the linear kernel (dot product) between two single vectors
        return np.dot(vector_1, vector_2)

    def compute_dual_objective(self, alphas, features, labels):
        # calculating the dual objective value by summing the quadratic and linear terms
        alpha_label_product = alphas * labels
        kernel_matrix = self.compute_linear_kernel(features, features)
        dual_term = 0.5 * np.dot(alpha_label_product, np.dot(kernel_matrix, alpha_label_product))
        return dual_term - np.sum(alphas)
    
    def predict(self, features):
        return [np.sign(np.dot(self.weights, feature) + self.bias) for feature in features]

def dual_svm(train_features, train_labels, test_features, test_labels):
    '''
    Trains and evaluates the dual svm model for different C values
    '''
    print("======== Running Dual SVM with SLSQP Optimization ========")
    dual_svm_model = DualSVM()

    for c in utilities.C:
        print(f"\n--- Evaluating for Hyperparameter C = {c:.4f} ---")

        # training the model
        dual_svm_model.train(train_features, train_labels, c)

        # displaying weights and bias
        print(f"  Weights: {dual_svm_model.weights}")
        print(f"  Bias: {dual_svm_model.bias}")

        # calculating training error
        train_predictions = dual_svm_model.predict(train_features)
        train_error = np.sum(train_predictions != train_labels)
        print(f"  Training Error: {train_error} ({(train_error / len(train_labels)) * 100:.2f}%)")

        # calculating testing error
        test_predictions = dual_svm_model.predict(test_features)
        test_error = np.sum(test_predictions != test_labels)
        print(f"  Testing Error: {test_error} ({(test_error / len(test_labels)) * 100:.2f}%)")

    print("\n======== Dual SVM Evaluation Complete ========")
