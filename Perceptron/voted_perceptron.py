import numpy as np
import utilities

class VotedPerceptron():
    def __init__(self, learning_rate=0.1, T=10, epoch_shuffled_indices=None):
        self.learning_rate = learning_rate
        self.T = T
        self._weight_vectors = [] # list of all weight vectors generated during training
        self._vote_counts = [] # corresponding count of correct predictions for each weight vector
        self._epoch_shuffled_indices = epoch_shuffled_indices
    
    def get_weight_vectors(self):
        weight_vectors_with_votes = []
        for i in range(len(self._vote_counts)):
            weight_vectors_with_votes.append((self._weight_vectors[i], self._vote_counts[i]))
        return weight_vectors_with_votes

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
        # clearing previous weight vectors and vote counts
        self._weight_vectors.clear()
        self._vote_counts.clear()

        current_weight_vector = np.zeros_like(features[0])
        vote_index = -1

        # setting shuffled indices for training order if provided, otherwise generating new ones
        shuffled_indices = self._epoch_shuffled_indices or self.generate_shuffled_indices(features, self.T)

        for epoch in range(self.T):
            for index in shuffled_indices[epoch]:
                # checking for misclassification and updating weight vector if necessary
                if labels[index] * np.dot(current_weight_vector, features[index]) <= 0:
                    # adding the current weight vector to the list before updating if not the initial vector
                    if vote_index != -1:
                        self._weight_vectors.append(np.copy(current_weight_vector))
                    # updating the current weight vector based on the misclassified example
                    current_weight_vector += self.learning_rate * labels[index] * features[index]
                    vote_index += 1
                    self._vote_counts.append(1)
                else:
                    # incrementing the vote count if the example is correctly classified
                    self._vote_counts[vote_index] += 1

        self._weight_vectors.append(np.copy(current_weight_vector))
    
    def predict_labels(self, features):
        # initializing a list to store predicted labels
        predicted_labels = []
        for feature_vector in features:
            weighted_vote_sum = 0  # initializing weighted vote sum for the current feature vector
            # getting weighted votes from each weight vector
            for i in range(len(self._vote_counts)):
                weighted_vote_sum += self._vote_counts[i] * np.sign(np.dot(feature_vector, self._weight_vectors[i]))
            predicted_labels.append(np.sign(weighted_vote_sum))
        return predicted_labels

def main():
    train_features, train_labels = utilities.read_csv(utilities.bank_note_train)
    test_features, test_labels = utilities.read_csv(utilities.bank_note_test)
    
    v_perceptron = VotedPerceptron(learning_rate=utilities.LEARNING_RATE, T=utilities.T)
    v_perceptron.train_perceptron(train_features, train_labels)

    # collecting distinct weight vectors list and their counts
    distinct_weights = {}
    for weight_vector, count in v_perceptron.get_weight_vectors():
        weight_tuple = tuple(weight_vector.flatten())  
        if weight_tuple not in distinct_weights:
            distinct_weights[weight_tuple] = count
        else:
            distinct_weights[weight_tuple] += count

    # displaying distinct weight vectors list and their counts
    print("\nVoted Perceptron Distinct Weight Vectors and Counts:")
    for i, (weight_vector, count) in enumerate(distinct_weights.items(), start=1):
        formatted_vector = [float(w) for w in weight_vector]
        print(f"Weight Vector {i}: {formatted_vector}")
        print(f"Count of Correct Classifications: {count}\n")

    # predictions 
    predictions = v_perceptron.predict_labels(test_features)

    #error calculation
    incorrect_count = np.sum(test_labels.flatten() != predictions)
    error_percentage = (incorrect_count / len(test_labels)) * 100
    
    # displaying average prediction error
    print("\nVoted Perceptron Test Results:")
    print(f"Incorrect Predictions: {incorrect_count} out of {len(test_labels)}")
    print(f"Average Prediction Error: {error_percentage:.3f}%")

    return 0

if __name__ == "__main__":
    main()
