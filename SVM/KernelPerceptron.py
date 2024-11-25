import numpy as np
import utilities
import GaussianDualSVM

MAX_ITERS = 50

class KernelPerceptron:
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn  # kernel fn 
        self.num_mistaks = None  # num of mistakes for each training example
        self.training_data = None  # store training data and labels
        self.training_labels = None

    def train(self, features, labels):
    #training the Kernel Perceptron using mistake counts
        num_samples = features.shape[0]
        self.training_data = features
        self.training_labels = labels
        self.num_mistaks = np.zeros(num_samples)

        for t in range(MAX_ITERS):
            for i in range(num_samples):
                # predicting labels
                prediction = self.predict_single(features[i])
                # if misclassified update the mistake count
                if prediction != labels[i]:
                    self.num_mistaks[i] += 1

    def predict_single(self, features):
    #predicting the label for a single input using the mistake counts
        dv = 0
        for i in range(len(self.num_mistaks)):
            if self.num_mistaks[i] > 0:
                dv += (
                    self.num_mistaks[i] * self.training_labels[i] * self.kernel_fn(self.training_data[i], features)
                )
        return np.sign(dv)

    def predict(self, features):
    #predicitng labels for a set of inputs
        predictions = []
        for x in features:
            predictions.append(self.predict_single(x))
        return np.array(predictions)

def kernel_perceptron(train_features, train_labels, test_features, test_labels):
    print("======== Running Kernel Perceptron with Gaussian Kernel ========")
    for gamma in utilities.LR:
        print(f"\n--- Evaluating for Gamma = {gamma} ---")
        
        # setting up Gaussian kernel with the current gamma
        gaussian_dual_svm = GaussianDualSVM.GaussianDualSVM()
        kernel_function = lambda feature_set_1, feature_set_2: gaussian_dual_svm.gaussian_kernel(feature_set_1, feature_set_2, gamma)
        
        # train the Kernel Perceptron
        kernel_perceptron = KernelPerceptron(kernel_function)
        kernel_perceptron.train(train_features, train_labels)

        # calculating training error
        train_predictions = kernel_perceptron.predict(train_features)
        train_error = np.sum(train_predictions != train_labels)
        train_error_percent = (train_error / len(train_labels)) * 100

        # calculating testing error
        test_predictions = kernel_perceptron.predict(test_features)
        test_error = np.sum(test_predictions != test_labels)
        test_error_percent = (test_error / len(test_labels)) * 100

        # Display results
        print(f"      Training Error: {train_error} ({train_error_percent:.2f}%)")
        print(f"      Testing Error: {test_error} ({test_error_percent:.2f}%)")
    print("======== Kernel Perceptron with Gaussian Kernel Complete ========")

