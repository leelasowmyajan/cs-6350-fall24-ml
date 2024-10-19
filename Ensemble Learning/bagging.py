from DecisionTree import id3_algo
import numpy as np
import utilities
import pandas as pd

class Bagging:
    def __init__(self, label, sub_sample_count, heuristic='entropy', debug=False) -> None:
        self._label = label
        self._heuristic = heuristic
        self._hypotheses = [] # all hypotheses (trees) generated from ID3
        self._sample_count = sub_sample_count # the number of examples to pull from the training data

        # Debug Properties
        self._debug = debug # if true, record each iter of _D and error

    def bag(self, examples, attributes, T=250):
        '''
            Run single_boost T number of times.
        '''
        
        for t in range(T):
            self.single_bag(examples, attributes)

    def single_bag(self, examples, attributes):
        '''
            Commit 1 additional tree to be bagged.
        '''

        # Get X number of rows, determined before entering tree creation
        sub_samples = examples.sample(n=self._sample_count, replace=True)

        # Find a classifier h_t whose weighted classification error is better than chance
        # >>> Just get the label with the best gain
        root = id3_algo.ID3_ALGO_EXTENDED(self._label, heuristic=self._heuristic).build_decision_tree(sub_samples.to_dict(orient="records"), attributes)
        
        # add it to the hypotheses list along with its weight (vote)
        self._hypotheses.append(root)

    def classify(self, inst):
        vote = 0

        for h in self._hypotheses:
            vote += int(h.predict(inst)) # will return either -1 or 1, no average needed
        
        # if > 0, we classify as 1
        # if < 0, we classify as -1
        return np.sign(vote)
    
    def verbose_classify(self, inst):
        vote = 0

        for i in range(len(self._hypotheses)):
            res = int(self._hypotheses[i].predict(inst))
            print("h (", str(self._hypotheses[i].label), ")", i, "says", str(res))
            vote += res
        
        return np.sign(vote)

    def get_hypothesis(self, index):
        return self._hypotheses[index]

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
    bagging_model = Bagging("y", sample_len, debug=utilities.debug_mode)

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
        bagging_model.single_bag(train_data, utilities.attributes)

        # Evaluate the model on the training set
        total_train_errors = sum([1 for i in range(num_train_samples) if train_data["y"][i] != bagging_model.classify(train_data.iloc[i])])
        training_error_percentage.append((total_train_errors / num_train_samples) * 100)
        training_error_total.append(total_train_errors)
        print(f"Training: Total Errors: {total_train_errors}/{num_train_samples} ({(total_train_errors / num_train_samples) * 100:.2f}%)")

        # Evaluate the model on the testing set
        total_test_errors = sum([1 for i in range(num_test_samples) if test_data["y"][i] != bagging_model.classify(test_data.iloc[i])])
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
