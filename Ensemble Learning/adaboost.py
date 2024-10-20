from DecisionTree import id3_algo
import numpy as np
import pandas as pd
import utilities

class AdaBoost:

    def __init__(self, target_label, num_samples, heuristic='entropy', debug_mode=False) -> None:
        self._target_label = target_label
        self._heuristic = heuristic
        self._learners_and_weights = [] 
        self._num_samples = num_samples
        self._debug_mode  = debug_mode
        self._weight_history  = [] 
        self._error_history = [] 

    def train_ensemble(self, data, features, num_iterations=250):
        actual_labels = data[self._target_label].tolist()
        for _ in range(num_iterations):
            self.train_single_learner(data, features, actual_labels)

    def train_single_learner(self, data, features, actual_labels):
        self.weights = data["weight"].tolist()
        stump = id3_algo.ID3_ALGO_EXTENDED(self._target_label, heuristic=self._heuristic).build_decision_tree_stump(data.to_dict(orient="records"), features)
        predictions = self.predict_with_learner(stump, data.to_dict(orient="records"))

        #calculating weighted error
        error = 0
        for i in range(self._num_samples):
            if predictions[i] != actual_labels[i]:
                error += self.weights[i]

        if self._debug_mode:
            self._error_history.append(error)
        vote_weight = 0.5 * np.log((1 - error) / error)        
        self._learners_and_weights.append((stump, vote_weight))

        n_factor = 0
        for i in range(self._num_samples):
            self.weights[i] = self.weights[i] * np.exp(-vote_weight * actual_labels[i] * predictions[i])
            n_factor += self.weights[i]

        for i in range(self._num_samples):
            self.weights[i] = self.weights[i] / n_factor

        data["weight"] = self.weights
        
    def predict_with_learner(self, learner, data):
        predict_results = []
        for i in range(self._num_samples):
            predict_results.append(int(learner.predict(data[i])))
        return predict_results

    def predict_with_ensemble(self, instance):
        weighted_sum = 0
        for learner, vote_weight in self._learners_and_weights:
            weighted_sum += vote_weight * int(learner.predict(instance))
        return np.sign(weighted_sum)
    
    def predict_with_specific_learner(self, data, index=0):
        return int(self._learners_and_weights[index][0].predict(data))
    
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
    adaboost_model = AdaBoost("y", num_train_samples, debug_mode=utilities.debug_mode)

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
        adaboost_model.train_single_learner(train_data, utilities.attributes, actual_train_labels)

        # Training Errors
        overall_train_error, stump_train_error = 0, 0
        for i in range(num_train_samples):
            if train_data["y"][i] != adaboost_model.predict_with_ensemble(train_data.iloc[i]):
                overall_train_error += 1
            if train_data["y"][i] != adaboost_model.predict_with_specific_learner(train_data.iloc[i], iteration):
                stump_train_error += 1

        overall_train_errors.append((overall_train_error / num_train_samples) * 100)
        stump_train_errors.append((stump_train_error / num_train_samples) * 100)
        
        print(f"Training Round {iteration + 1}: Overall Error: {overall_train_error}/{num_train_samples} ({(overall_train_error / num_train_samples) * 100:.2f}%)")
        print(f"Training Round {iteration + 1}: Stump Error: {stump_train_error}/{num_train_samples} ({(stump_train_error / num_train_samples) * 100:.2f}%)")

        # Testing Errors
        overall_test_error, stump_test_error = 0, 0
        for i in range(num_test_samples):
            if test_data["y"][i] != adaboost_model.predict_with_ensemble(test_data.iloc[i]):
                overall_test_error += 1
            if test_data["y"][i] != adaboost_model.predict_with_specific_learner(test_data.iloc[i], iteration):
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