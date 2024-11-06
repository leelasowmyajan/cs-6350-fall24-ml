import numpy as np
import utilities

class StandardPerceptron():
    def __init__(self, learning_rate=0.1, T=10, epoch_shuffled_indices=None):
        self.learning_rate = learning_rate
        self.T = T
        self._weight_vector = None
        self._epoch_shuffled_indices = epoch_shuffled_indices

    def get_weight_vector(self):
        return self._weight_vector

    def generate_shuffled_indices(self, data, num_epochs):
        # creating a list to store shuffled indices for each epoch
        epoch_shuffled_indices = []
        num_samples = data.shape[0]
        for _ in range(num_epochs):
            # generating a sequence of indices and shuffling it randomly
            indices = np.arange(num_samples)
            np.random.default_rng().shuffle(indices)
            epoch_shuffled_indices.append(indices)
        return epoch_shuffled_indices

    def train_perceptron(self, features, labels):
        # initializing weight vector with zeros
        self._weight_vector = np.zeros_like(features[0])

        # setting shuffled indices for training order if provided, otherwise generating new ones
        shuffled_indices = self._epoch_shuffled_indices or self.generate_shuffled_indices(features, self.T)

        for epoch in range(self.T):
            for index in shuffled_indices[epoch]:
                # updating weight vector if there is a misclassification
                if labels[index] * np.dot(self._weight_vector, features[index]) <= 0:
                    self._weight_vector += self.learning_rate * labels[index] * features[index]
            
    def predict_labels(self, features):
        # initializing a list to store predicted labels
        predicted_labels = []
        # calculating the predicted label for each feature vector
        for feature_vector in features:
            predicted_labels.append(np.sign(np.dot(feature_vector, self._weight_vector)))
        return predicted_labels

def main():
    train_features, train_labels = utilities.read_csv(utilities.bank_note_train)
    test_features, test_labels = utilities.read_csv(utilities.bank_note_test)
    
    s_perceptron = StandardPerceptron(learning_rate=utilities.LEARNING_RATE, T=utilities.T)
    s_perceptron.train_perceptron(train_features, train_labels)

    # displaying the learned weight vector
    learned_weights = s_perceptron.get_weight_vector()
    print("\nStandard Perceptron Learned Weight Vector:")
    print(learned_weights)

    # predictions and error calculation
    predictions = s_perceptron.predict_labels(test_features)
    incorrect_count = np.sum(test_labels.flatten() != predictions)
    error_percentage = (incorrect_count / len(test_labels)) * 100
    # displaying average prediction error
    print("\nStandard Perceptron Test Results:")
    print(f"Incorrect Predictions: {incorrect_count} out of {len(test_labels)}")
    print(f"Average Prediction Error: {error_percentage:.2f}%")
    
    return 0

if __name__ == "__main__":
    main()
