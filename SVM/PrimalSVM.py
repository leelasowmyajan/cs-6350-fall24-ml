import numpy as np
import utilities

def lr_sched_one(intial_lr, lr_decay, epoch):
    return intial_lr / (1 + intial_lr / lr_decay * epoch)
    
def lr_sched_two(inital_lr, lr_decay, epoch):
    return inital_lr / (1 + epoch)

class PrimalSVM():
    def __init__(self, learning_rate_schedule):
        self.weights = None  # stores the model weights
        self.num_updates = 0  # keeps track of the number of updates made
        self.learning_rate_schedule = learning_rate_schedule
    
    def generate_shuffled_indices(self, data, T):
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

    def train(self, features, labels, C, intial_lr, lr_decay, T=utilities.T):
        """
        Trains the SVM model using stochastic sub-gradient descent.
        """
        # adding bias term to the features by appending a column of ones
        feat_with_bias = np.insert(features, features.shape[1], [1]*features.shape[0], axis=1)

        # generating shuffled indices for each epoch
        shuffled_indices_per_epoch = self.generate_shuffled_indices(features, T)

        num_samples = features.shape[0]

        # initializing the weight vector as zeros with the appropriate dimension
        self.weights = np.zeros((feat_with_bias.shape[1],))

        for epoch in range(T):
            # calculating the learning rate for the current epoch
            lr = self.learning_rate_schedule(intial_lr, lr_decay, epoch)
            
            for i in shuffled_indices_per_epoch[epoch]:
                if labels[i] * np.dot(feat_with_bias[i], self.weights) <= 1:
                    # updating the weights when loss is violated
                    self.weights = (1 - lr) * self.weights + lr * C * num_samples * labels[i] * feat_with_bias[i]
                else:
                    # applying regularization to the weights when loss is not violated
                    self.weights *= (1 - lr)
    
    def predict(self, features):
        """
        Predicts labels for a given set of feature vectors.
        """
        # adding bias as an additional feature
        feat_with_bias = np.insert(features, features.shape[1], [1] * features.shape[0], axis=1)
        return np.sign(np.dot(feat_with_bias, self.weights))

def primal_svm(train_features, train_labels, test_features, test_labels):
    print("======== Running Primal SVM with Stochastic Sub-Gradient Descent ========")

    # initializing the SVM models with different learning rate schedules
    psvm_schedule_one = PrimalSVM(lr_sched_one)
    psvm_schedule_two = PrimalSVM(lr_sched_two)

    for c in utilities.C:
        print(f"\n--- Evaluating for Hyperparameter C = {c:.4f} ---")

        avg_train_error_sched_one = 0
        avg_test_error_sched_one = 0
        avg_train_error_sched_two = 0
        avg_test_error_sched_two = 0

        total_weights_sched_one = np.zeros((train_features.shape[1] + 1,))
        total_weights_sched_two = np.zeros((train_features.shape[1] + 1,))
        
        for _ in range(10):
            # training SVMs with each schedule
            psvm_schedule_one.train(train_features, train_labels, c, utilities.INITIAL_LR, utilities.LR_DECAY)
            psvm_schedule_two.train(train_features, train_labels, c, utilities.INITIAL_LR, utilities.LR_DECAY)

            # accumulating weights for averaging
            total_weights_sched_one += psvm_schedule_one.weights
            total_weights_sched_two += psvm_schedule_two.weights

            # calculating training errors for schedule one
            train_predictions_sched_one = psvm_schedule_one.predict(train_features)
            avg_train_error_sched_one += np.sum(train_predictions_sched_one != train_labels)

            # calculating test errors for schedule one
            test_predictions_sched_one = psvm_schedule_one.predict(test_features)
            avg_test_error_sched_one += np.sum(test_predictions_sched_one != test_labels)

            # calculating training errors for schedule two
            train_predictions_sched_two = psvm_schedule_two.predict(train_features)
            avg_train_error_sched_two += np.sum(train_predictions_sched_two != train_labels)

            # calculating test errors for schedule two
            test_predictions_sched_two = psvm_schedule_two.predict(test_features)
            avg_test_error_sched_two += np.sum(test_predictions_sched_two != test_labels)

        
        # averaging weights and errors over the 10 runs
        avg_weights_sched_one = total_weights_sched_one / 10
        avg_weights_sched_two = total_weights_sched_two / 10

        avg_train_error_sched_one /= 10
        avg_test_error_sched_one /= 10
        avg_train_error_sched_two /= 10
        avg_test_error_sched_two /= 10
        
        # displaying results for schedule one
        print("\nLearning Rate Schedule 1")
        print(f"  Average Weights: {avg_weights_sched_one}")
        print(f"  Training Error: {avg_train_error_sched_one} ({(avg_train_error_sched_one / len(train_labels)) * 100:.2f}%)")
        print(f"  Testing Error: {avg_test_error_sched_one} ({(avg_test_error_sched_one / len(test_labels)) * 100:.2f}%)")

        # displaying results for schedule two
        print("\nLearning Rate Schedule 2")
        print(f"  Average Weights: {avg_weights_sched_two}")
        print(f"  Training Error: {avg_train_error_sched_two} ({(avg_train_error_sched_two / len(train_labels)) * 100:.2f}%)")
        print(f"  Testing Error: {avg_test_error_sched_two} ({(avg_test_error_sched_two / len(test_labels)) * 100:.2f}%)")

    print("\n======== Primal SVM Evaluation Complete ========")