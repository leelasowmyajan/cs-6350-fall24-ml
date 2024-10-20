from DecisionTree import id3_algo
import numpy as np
import utilities
import pandas as pd

class Bagging:
    def __init__(self, target_label, sample_size, heuristic='entropy', debug_mode=False) -> None:
        self._target_label = target_label
        self._heuristic = heuristic
        self._learners = [] 
        self._sample_size = sample_size 
        self._debug_mode = debug_mode 

    def train_ensemble(self, data, features, num_iterations=250):
        for _ in range(num_iterations):
            self.train_single_learner(data, features)

    def train_single_learner(self, data, features):
        sub_samples = data.sample(n=self._sample_size, replace=True)
        root = id3_algo.ID3_ALGO_EXTENDED(self._target_label, heuristic=self._heuristic).build_decision_tree(sub_samples.to_dict(orient="records"), features)        
        self._learners.append(root)

    def predict_with_ensemble(self, instance):
        vote_sum = 0
        for learner in self._learners: 
            vote_sum += int(learner.predict(instance))
        return np.sign(vote_sum)

    def get_learner(self, index):
        return self._learners[index]

def main():
    print("Starting Bagging Tests...")

    # Load the training and testing datasets
    train_data = utilities.read_csv(utilities.bank_train)
    num_train_samples = len(train_data.index)
    sample_len = int(utilities.sample_perc * num_train_samples)  # Percentage of training samples used for each bagging round

    # For Bagging, assign uniform weight to all samples
    train_data["weight"] = 1

    test_data = utilities.read_csv(utilities.bank_test)
    num_test_samples = len(test_data.index)

    # Initialize Bagging model
    bagging_model = Bagging("y", sample_len, debug_mode=utilities.debug_mode)

    # Initialize lists to store errors over iterations
    training_error_percentage = []
    training_error_total = []
    testing_error_percentage = []
    testing_error_total = []

    # List to track the number of iterations
    iterations = list(range(utilities.num_iterations))

    # Loop through the bagging iterations
    for iteration in iterations:
        print(f"\n=== Bagging Iteration {iteration + 1} ===")

        # Train the Bagging model
        bagging_model.train_single_learner(train_data, utilities.attributes)

        # Evaluate the model on the training set
        total_train_errors = sum([1 for i in range(num_train_samples) if train_data["y"][i] != bagging_model.predict_with_ensemble(train_data.iloc[i])])
        training_error_percentage.append((total_train_errors / num_train_samples) * 100)
        training_error_total.append(total_train_errors)
        print(f"Training: Total Errors: {total_train_errors}/{num_train_samples} ({(total_train_errors / num_train_samples) * 100:.2f}%)")

        # Evaluate the model on the testing set
        total_test_errors = sum([1 for i in range(num_test_samples) if test_data["y"][i] != bagging_model.predict_with_ensemble(test_data.iloc[i])])
        testing_error_percentage.append((total_test_errors / num_test_samples) * 100)
        testing_error_total.append(total_test_errors)
        print(f"Testing: Total Errors: {total_test_errors}/{num_test_samples} ({(total_test_errors / num_test_samples) * 100:.2f}%)")

    # Store the results in a DataFrame
    results_df = pd.DataFrame({
        "Iteration": iterations,
        "Training Error (%)": training_error_percentage,
        "Testing Error (%)": testing_error_percentage,
    })

    # Save the results to a CSV file
    results_df.to_csv(utilities.results_path + "bagging_results.csv", index=False)

    print("Finished Bagging Tests...")
    #print("\nResults Summary:\n", results_df)

if __name__ == "__main__":
    main()
