from DecisionTree import id3_algo
import numpy as np
import pandas as pd
import utilities

class AdaBoost:

    def __init__(self, label, sample_count, heuristic='entropy', debug=False) -> None:
        self._label = label
        self._heuristic = heuristic
        self._hypotheses = [] # entries take the form of a tuple: (ID3 tree, vote_weight)
        self._sample_count = sample_count

        # Debug Properties
        self._debug  = debug # if true, record each iter of _D and error
        self._prevD  = [] # a list of lists with the full list being equal to T and each sublist being equal to size len(self._D) or sample_count
        self._errors = [] # should always equal len(self._hypotheses)

    def boost(self, examples, attributes, T=250):
        '''
            Run single_boost T number of times.
        '''
        actual_labels = examples[self._label].tolist()
        
        # for t = 1, 2, ..., T
        for t in range(T):
            self.single_boost(examples, attributes, actual_labels)

    def single_boost(self, examples, attributes, actual_labels):

        self.weights = examples["weight"].tolist()

        root = id3_algo.ID3_ALGO_EXTENDED(self._label, heuristic=self._heuristic).build_decision_tree_stump(examples.to_dict(orient="records"), attributes)
        
        # What does the tree predict?
        predictions = self.classify_single_h(root, examples.to_dict(orient="records"))

        # What is the weighted classification error based on this tree?
        error = self._compute_error(predictions, actual_labels)

        if self._debug:
            self._errors.append(error)

        # Compute its vote
        alpha = self._compute_vote(error)
        
        # add it to the hypotheses list along with its weight (vote)
        self._hypotheses.append((root, alpha))

        # Update the values of the weights for the training examples
        self._update_weights(alpha, actual_labels, predictions)

        # Update weights in panda
        examples["weight"] = self.weights

    def _compute_vote(self, error):
        return 0.5 * np.log((1 - error) / error)
    
    def _compute_error(self, y, x):
        error = 0

        for i in range(self._sample_count):
            if y[i] != x[i]:
                error += self.weights[i]

        return error
    
    def _update_weights(self, alpha, actual_labels, predictions):
        Z = 0
        for i in range(self._sample_count):
            self.weights[i] = self.weights[i] * np.exp(-alpha * actual_labels[i] * predictions[i])
            Z += self.weights[i]

        for i in range(self._sample_count):
            self.weights[i] = self.weights[i] / Z
    
    def classify_single_h(self, hypothesis, data):
        pred_results = []
        
        for i in range(self._sample_count):
            pred_results.append(int(hypothesis.predict(data[i])))
        
        return pred_results

    def classify(self, inst):
        wsum = 0

        for h, alpha in self._hypotheses:
            wsum += alpha * int(h.predict(inst))
        
        return np.sign(wsum)
    
    def test_hypothesis_at(self, data, index=0):
        return int(self._hypotheses[index][0].predict(data))
    
    def verbose_classify(self, inst):
        wsum = 0

        for i in range(len(self._hypotheses)):
            res = int(self._hypotheses[i][0].predict(inst))
            print("h (", str(self._hypotheses[i][1]), ")", i, "says", str(res))
            wsum += self._hypotheses[i][1] * res
        
        return np.sign(wsum)
    
def main():
    print("Starting AdaBoost tests...")

    # Load training and testing data
    train_data = utilities.read_csv(utilities.bank_train)
    num_train_samples = len(train_data.index)

    # Initialize the weight column with equal weight for all training examples
    train_data["weight"] = 1 / num_train_samples

    test_data = utilities.read_csv(utilities.bank_test)
    num_test_samples = len(test_data.index)

    # Initialize AdaBoost model
    adaboost_model = AdaBoost("y", num_train_samples, debug=utilities.debug_mode)

    # Actual labels for training data
    actual_train_labels = train_data["y"].tolist()

    # Lists to store errors over iterations
    overall_train_errors = []
    overall_test_errors = []
    stump_train_errors = []
    stump_test_errors = []

    # Loop through each boosting iteration
    for iteration in range(utilities.num_iterations):
        print(f"\n=== Boosting Round {iteration + 1} ===")
        
        # Train the AdaBoost model
        adaboost_model.single_boost(train_data, utilities.attributes, actual_train_labels)

        # Training Errors
        overall_train_error, stump_train_error = 0, 0
        for i in range(num_train_samples):
            if train_data["y"][i] != adaboost_model.classify(train_data.iloc[i]):
                overall_train_error += 1
            if train_data["y"][i] != adaboost_model.test_hypothesis_at(train_data.iloc[i], iteration):
                stump_train_error += 1

        overall_train_errors.append((overall_train_error / num_train_samples) * 100)
        stump_train_errors.append((stump_train_error / num_train_samples) * 100)
        
        print(f"Training Round {iteration + 1}: Overall Error: {overall_train_error}/{num_train_samples} ({(overall_train_error / num_train_samples) * 100:.2f}%)")
        print(f"Training Round {iteration + 1}: Stump Error: {stump_train_error}/{num_train_samples} ({(stump_train_error / num_train_samples) * 100:.2f}%)")

        # Testing Errors
        overall_test_error, stump_test_error = 0, 0
        for i in range(num_test_samples):
            if test_data["y"][i] != adaboost_model.classify(test_data.iloc[i]):
                overall_test_error += 1
            if test_data["y"][i] != adaboost_model.test_hypothesis_at(test_data.iloc[i], iteration):
                stump_test_error += 1

        overall_test_errors.append((overall_test_error / num_test_samples) * 100)
        stump_test_errors.append((stump_test_error / num_test_samples) * 100)
        
        print(f"Testing Round {iteration + 1}: Overall Error: {overall_test_error}/{num_test_samples} ({(overall_test_error / num_test_samples) * 100:.2f}%)")
        print(f"Testing Round {iteration + 1}: Stump Error: {stump_test_error}/{num_test_samples} ({(stump_test_error / num_test_samples) * 100:.2f}%)")

    # Storing results in a DataFrame
    results_df = pd.DataFrame({
        "Iteration": list(range(utilities.num_iterations)),
        "Training Error (%)": overall_train_errors,
        "Testing Error (%)": overall_test_errors
    })

    stump_results_df = pd.DataFrame({
        "Iteration": list(range(utilities.num_iterations)),
        "Training Stump Error (%)": stump_train_errors,
        "Testing Stump Error (%)": stump_test_errors
    })

    # Saving results to CSV files
    results_df.to_csv(utilities.results_path + "adaboost_overall_results.csv", index=False)
    stump_results_df.to_csv(utilities.results_path + "adaboost_stump_results.csv", index=False)

    print("Finished AdaBoost tests...")
    # print("\nOverall Results:\n", results_df)
    # print("\nStump Results:\n", stump_results_df)

if __name__ == "__main__":
    main()