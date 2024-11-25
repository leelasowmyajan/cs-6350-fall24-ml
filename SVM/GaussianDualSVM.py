import numpy as np
from scipy.optimize import minimize
import utilities

class GaussianDualSVM:
    def __init__(self):
        self.support_vector_indices = [] # stores indices of support vectors
        self.bias = 0
        self.threshold = 1e-10
        self.alphas = None # stores dual variables (alphas)

    def train(self, features, labels, C, gamma):
        """
        Trains the SVM model using the dual formulation with a Gaussian kernel.
        """
        num_samples = features.shape[0]
        initial_alphas = np.zeros(num_samples)

        # solving the optimization problem
        result = minimize(
            self.compute_gaussian_objective,
            initial_alphas,
            args=(features, labels, gamma),
            method='SLSQP',
            constraints=[{'type': 'eq', 'fun': lambda alphas: np.dot(alphas, labels)}],
            bounds=[(0, C)] * num_samples
        )

        self.alphas = result.x

        # identifying support vectors
        self.support_vector_indices = [i for i, alpha in enumerate(self.alphas) if alpha > self.threshold]

        # computing bias term
        support_kernel = self.compute_gaussian_kernel(features[self.support_vector_indices], features, gamma)
        self.bias = np.mean(
            labels[self.support_vector_indices] - np.dot(support_kernel, self.alphas * labels)
        )

    def compute_gaussian_objective(self, alphas, features, labels, gamma):
        """
        Computes the dual objective function value with the Gaussian kernel.
        """
        kernel_matrix = self.compute_gaussian_kernel(features, features, gamma)
        term_1 = 0.5 * np.sum(np.outer(alphas, alphas) * kernel_matrix * np.outer(labels, labels))
        term_2 = np.sum(alphas)
        return term_1 - term_2

    def compute_gaussian_kernel(self, feature_set_1, feature_set_2, gamma):
        """
        Computes the Gaussian kernel between two sets of feature vectors. made this more efficient. 
        """
        sq_dists = (
            np.sum(feature_set_1**2, axis=1)[:, np.newaxis] + 
            np.sum(feature_set_2**2, axis=1) - 
            2 * np.dot(feature_set_1, feature_set_2.T)
        )
        return np.exp(-sq_dists / gamma)
    
    def gaussian_kernel(self, feature_set_1, feature_set_2, gamma):
        """
        Compute the Gaussian kernel between two feature vectors.
        """
        sq_dist = np.sum((feature_set_1 - feature_set_2) ** 2)
        return np.exp(-sq_dist / gamma)

    def predict(self, features, train_features, train_labels, gamma):
        """
        Predicts the labels for a given set of features.
        """
        kernel_matrix = self.compute_gaussian_kernel(features, train_features, gamma)
        decision_values = np.dot(kernel_matrix, self.alphas * train_labels) + self.bias
        return np.sign(decision_values)


def gaussian_dual_svm(train_features, train_labels, test_features, test_labels):
    print("======== Running Gaussian Dual SVM with SLSQP Optimization ========")
    gaussian_dual_svm = GaussianDualSVM()

    best_combination = None
    best_test_error = float('inf')

    for gamma in utilities.LR:
        print(f"\n--- Evaluating for Gamma = {gamma} ---")
        
        for c in utilities.C:
            print(f"\n   === Evaluating for Hyperparameter C = {c:.4f} ===")

            gaussian_dual_svm.train(train_features, train_labels, c, gamma)

            print(f"      Bias: {gaussian_dual_svm.bias}")
            print(f"      Number of Support Vectors: {len(gaussian_dual_svm.support_vector_indices)}")

            # calculating training error
            train_predictions = gaussian_dual_svm.predict(train_features, train_features, train_labels, gamma)
            train_error = np.sum(train_predictions != train_labels)
            train_error_percent = (train_error / len(train_labels)) * 100

            # calculating testing error
            test_predictions = gaussian_dual_svm.predict(test_features, train_features, train_labels, gamma)
            test_error = np.sum(test_predictions != test_labels)
            test_error_percent = (test_error / len(test_labels)) * 100

            # Display results
            print(f"      Training Error: {train_error} ({train_error_percent:.2f}%)")
            print(f"      Testing Error: {test_error} ({test_error_percent:.2f}%)")

            # Track best combination
            if test_error < best_test_error:
                best_test_error = test_error_percent
                best_combination = (gamma, c)
    
    print(f"\nBest Combination: Gamma = {best_combination[0]}, C = {best_combination[1]}, Testing Error = {test_error_percent:.2f}%")
    print("\n======== Gaussian Dual SVM Evaluation Complete ========")

def overlap(train_features, train_labels):
    print("======== Evaluating Overlap of Support Vectors ========")
    gaussian_dual_svm = GaussianDualSVM()

    # initializing dict to store support vector indices for each gamma
    gamma_support_vectors = {gamma: None for gamma in utilities.LR}

    # training SVM for each gamma and store support vector indices
    for gamma in utilities.LR:
        gaussian_dual_svm.train(train_features, train_labels, utilities.C_OVERLAP, gamma)
        gamma_support_vectors[gamma] = gaussian_dual_svm.support_vector_indices.copy()

    # reporting overlaps between consecutive gammas
    overlaps = {}
    gamma_list = list(utilities.LR)
    for g1, g2 in zip(gamma_list[:-1], gamma_list[1:]):
        overlaps[(g1, g2)] = len(np.intersect1d(gamma_support_vectors[g1], gamma_support_vectors[g2]))
    
    # displaying results
    for (g1, g2), count in overlaps.items():
        print(f"Overlap between Gamma = {g1} and Gamma = {g2}: {count} support vectors")
    
    print("======== Overlap Evaluation Complete ========")
