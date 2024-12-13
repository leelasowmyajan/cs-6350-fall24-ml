import numpy as np
import utilities

import warnings
warnings.filterwarnings("ignore")

class LogisticRegression():
    def __init__(self, num_weights, estimation_method="mle") -> None:
        # selecting loss and gradient functions based on estimation method
        method_mapping = {
            "mle": (self.mle_loss_function, self.mle_gradient), 
            "map": (self.map_loss_function, self.map_gradient)
            }
        self.loss_function, self.gradient_function = method_mapping[estimation_method]
        # initializing weights as zeros
        self.weights = np.zeros(num_weights)

    def train_logreg(self, features, labels, prior_variance = 0, decay_rate = 0.1, learning_rate = 0.01, T=100, printOp=False):
        num_samples = len(features)
        shuffled_indices = utilities.generate_shuffled_indices(features, T)
        if printOp:
            print("Shuffled indices for each epoch:\n", shuffled_indices)
        for t in range(T):
            current_lr = self.lr_sched(t, learning_rate, decay_rate)
            for sample_idx in shuffled_indices[t]:
                sample_features = features[sample_idx]
                sample_label = labels[sample_idx]
                gradient = self.gradient_function(sample_features, sample_label, num_samples, prior_variance)
                if printOp:
                    print(f"Sample index: {sample_idx}, Gradient: {gradient}")
                # updating weights using calculated gradient
                self.weights = self.weights - current_lr * gradient

    def lr_sched(self, t, intial_lr, lr_decay):
        return intial_lr / (1 + intial_lr / lr_decay * t)

    def mle_loss_function(self, features, label, _):
        # calculating the MLE loss
        linear_comb = np.dot(self.weights, features)
        logistic_loss = np.log(1 + np.exp(-label * linear_comb))
        return logistic_loss

    def mle_gradient(self, features, label, num_samples, x):
        # calculating the gradient for MLE

        # calculating the dot product of weights and features, multiplied by the label
        pdt = label * np.dot(self.weights, features)
        
        # calculating the logistic derivative using the sigmoid function
        logistic_derivative = 1 / (1 + np.exp(pdt))
        
        # calculating and returning the negative gradient scaled by the number of samples
        return -num_samples * logistic_derivative * label * features

    def map_loss_function(self, features, label, prior_variance):
        # calculating the MAP loss
        linear_comb = np.dot(self.weights, features)
        logistic_loss = np.log(1 + np.exp(-label * linear_comb))
        regularization_term = prior_variance * np.dot(self.weights, self.weights)
        return logistic_loss + regularization_term

    def map_gradient(self, features, label, num_samples, prior_variance):
        # calculating the gradient for MAP
        pdt = label * np.dot(self.weights, features)
        logistic_derivative = 1 + np.exp(pdt)
        return -num_samples * features * label / logistic_derivative + self.weights / prior_variance

    def predict_batch(self, feature_set):
        # predicting labels for a set of features
        return [self.predict_single(feature_vector) for feature_vector in feature_set]

    def predict_single(self, features):
        # predicting label for a single feature vector
        return -1 if np.dot(self.weights, features) <= 0 else 1

def main():
    print("========== Logistic Regression Implementation ==========\n")
    # defining sample data for problem 4 
    p4_features = np.array([[0.5, -1, 0.3, 1], [-1, -2, -2, 1], [1.5, 0.2, -2.5, 1]])
    p4_labels = np.array([[1], [-1], [1]])

    print("========== MAP ESTIMATION CHECK ==========")
    print("Testing MAP estimation with Problem 4 example...")
    # initializing logistic regression model for MAP estimation
    logistic_reg_map = LogisticRegression(len(p4_features[0]), estimation_method="map")
    logistic_reg_map.train_logreg(p4_features, p4_labels, learning_rate=0.0025, prior_variance=3, T=5, printOp=True)
    print("MAP estimation test completed successfully.\n")

    # loading training and testing datasets
    train_features, train_labels = utilities.read_csv(utilities.bank_note_train)
    test_features, test_labels = utilities.read_csv(utilities.bank_note_test)

    # calculating total combinations of variances and methods (MLE and MAP)
    total_combinations = len(utilities.P_VARIANCES) * 2

    print(f"Starting testing with {total_combinations} configurations...")

    # iterating over each variance value
    for variance in utilities.P_VARIANCES:
        print(f"Testing with variance={variance}")

        # initializing and training logistic regression model for MLE estimation
        mle_model = LogisticRegression(len(train_features[0]), estimation_method="mle")
        mle_model.train_logreg(
            train_features, train_labels,
            decay_rate=utilities.DECAY_RATE, learning_rate=utilities.INITIAL_LEARNING_RATE, T=utilities.T
        )

        # calculating training and testing errors for MLE
        mle_train_errors = sum(
            1 for pred, actual in zip(mle_model.predict_batch(train_features), train_labels)
            if pred != actual
        )
        mle_test_errors = sum(
            1 for pred, actual in zip(mle_model.predict_batch(test_features), test_labels)
            if pred != actual
        )

        print(f"\tMLE Training Error: {mle_train_errors / len(train_features) * 100:.4f}%")
        print(f"\tMLE Testing Error: {mle_test_errors / len(test_features) * 100:.4f}%")

        # initializing and training logistic regression model for MAP estimation
        map_model = LogisticRegression(len(train_features[0]), estimation_method="map")
        map_model.train_logreg(
            train_features, train_labels,
            decay_rate=utilities.DECAY_RATE, learning_rate=utilities.INITIAL_LEARNING_RATE, prior_variance=variance, T=utilities.T
        )

        # calculating training and testing errors for MAP
        map_train_errors = sum(
            1 for pred, actual in zip(map_model.predict_batch(train_features), train_labels)
            if pred != actual
        )
        map_test_errors = sum(
            1 for pred, actual in zip(map_model.predict_batch(test_features), test_labels)
            if pred != actual
        )

        print(f"\tMAP Training Error: {map_train_errors / len(train_features) * 100:.4f}%")
        print(f"\tMAP Testing Error: {map_test_errors / len(test_features) * 100:.4f}%")

    print(f"Finished testing with {total_combinations} configurations.")
    print("========== Logistic Regression Implementation Completed ==========")

if __name__ == "__main__":
    main()