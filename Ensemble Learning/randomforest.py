from DecisionTree import id3_algo
import pandas as pd
import numpy as np
import utilities

class RandomForest:
    def __init__(self, label, sub_sample_count, sub_attrib_count, heuristic='entropy', debug=False) -> None:
        self._label = label
        self._heuristic = heuristic
        self._hypotheses = [] # all hypotheses (trees) generated from ID3
        self._sample_count = sub_sample_count # the number of examples to pull from the training data
        self._attrib_count = sub_attrib_count # the number of attributes to go into G from A

        # Debug Properties
        self._debug = debug # if true, record each iter of _D and error

    def random_forest(self, examples, attributes, T=250):
        '''
            Run single_boost T number of times.
        '''
        
        for t in range(T):
            self.single_tree(examples, attributes)

    def single_tree(self, examples, attributes):
        '''
            Commit 1 additional tree to the random forest.
        '''

        # Get X number of rows, determined before entering tree creation
        sub_samples = examples.sample(n=self._sample_count, replace=True)
        
        # Find a classifier h_t whose weighted classification error is better than chance
        # >>> Just get the label with the best gain
        root = id3_algo.ID3_ALGO_EXTENDED(self._label, heuristic=self._heuristic).build_decision_tree_with_subset(sub_samples.to_dict(orient="records"), attributes, self._attrib_count)
        
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
    print("Starting Random Forest Tests...")

    # Load the training and testing datasets
    train_data = utilities.read_csv(utilities.bank_train)
    num_train_samples = len(train_data.index)
    sample_size = int(utilities.sample_perc * num_train_samples)  # Percentage of training samples used for each forest round

    # Assign uniform weight to all samples
    train_data["weight"] = 1  

    test_data = utilities.read_csv(utilities.bank_test)
    num_test_samples = len(test_data.index)

    # Initialize lists for storing errors
    rf2_train_error_percentage, rf2_train_error_total = [], []
    rf2_test_error_percentage, rf2_test_error_total = [], []

    rf4_train_error_percentage, rf4_train_error_total = [], []
    rf4_test_error_percentage, rf4_test_error_total = [], []

    rf6_train_error_percentage, rf6_train_error_total = [], []
    rf6_test_error_percentage, rf6_test_error_total = [], []

    iterations = list(range(utilities.num_iterations))

    # Initialize Random Forest models with different numbers of features
    rf2_model = RandomForest("y", sample_size, 2, debug=utilities.debug_mode)
    rf4_model = RandomForest("y", sample_size, 4, debug=utilities.debug_mode)
    rf6_model = RandomForest("y", sample_size, 6, debug=utilities.debug_mode)
    
    # Run the random forest tests for each iteration
    for iteration in iterations:
        print(f"\n=== Iteration {iteration + 1} ===")

        # Train each Random Forest model with the specified number of features
        rf2_model.single_tree(train_data, utilities.attributes)
        rf4_model.single_tree(train_data, utilities.attributes)
        rf6_model.single_tree(train_data, utilities.attributes)

        # Evaluate models on the training set
        tr_rf2_wrong = sum([1 for i in range(num_train_samples) if train_data["y"][i] != rf2_model.classify(train_data.iloc[i])])
        tr_rf4_wrong = sum([1 for i in range(num_train_samples) if train_data["y"][i] != rf4_model.classify(train_data.iloc[i])])
        tr_rf6_wrong = sum([1 for i in range(num_train_samples) if train_data["y"][i] != rf6_model.classify(train_data.iloc[i])])

        rf2_train_error_percentage.append(tr_rf2_wrong / num_train_samples * 100)
        rf2_train_error_total.append(tr_rf2_wrong)

        rf4_train_error_percentage.append(tr_rf4_wrong / num_train_samples * 100)
        rf4_train_error_total.append(tr_rf4_wrong)

        rf6_train_error_percentage.append(tr_rf6_wrong / num_train_samples * 100)
        rf6_train_error_total.append(tr_rf6_wrong)

        # Evaluate models on the testing set
        te_rf2_wrong = sum([1 for i in range(num_test_samples) if test_data["y"][i] != rf2_model.classify(test_data.iloc[i])])
        te_rf4_wrong = sum([1 for i in range(num_test_samples) if test_data["y"][i] != rf4_model.classify(test_data.iloc[i])])
        te_rf6_wrong = sum([1 for i in range(num_test_samples) if test_data["y"][i] != rf6_model.classify(test_data.iloc[i])])

        rf2_test_error_percentage.append(te_rf2_wrong / num_test_samples * 100)
        rf2_test_error_total.append(te_rf2_wrong)

        rf4_test_error_percentage.append(te_rf4_wrong / num_test_samples * 100)
        rf4_test_error_total.append(te_rf4_wrong)

        rf6_test_error_percentage.append(te_rf6_wrong / num_test_samples * 100)
        rf6_test_error_total.append(te_rf6_wrong)

        # Print errors for the current iteration
        print(f"[{iteration + 1} | 2 Features] TRAINING: {tr_rf2_wrong}/{num_train_samples} ({tr_rf2_wrong / num_train_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 4 Features] TRAINING: {tr_rf4_wrong}/{num_train_samples} ({tr_rf4_wrong / num_train_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 6 Features] TRAINING: {tr_rf6_wrong}/{num_train_samples} ({tr_rf6_wrong / num_train_samples * 100:.2f}%)")

        print(f"[{iteration + 1} | 2 Features] TESTING: {te_rf2_wrong}/{num_test_samples} ({te_rf2_wrong / num_test_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 4 Features] TESTING: {te_rf4_wrong}/{num_test_samples} ({te_rf4_wrong / num_test_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 6 Features] TESTING: {te_rf6_wrong}/{num_test_samples} ({te_rf6_wrong / num_test_samples * 100:.2f}%)")

    # Store the results in a DataFrame
    results_df = pd.DataFrame({
        "Iteration": iterations,
        "2 Features Training Error (%)": rf2_train_error_percentage,
        "2 Features Testing Error (%)": rf2_test_error_percentage,
        "4 Features Training Error (%)": rf4_train_error_percentage,
        "4 Features Training Errors": rf4_train_error_total,
        "4 Features Testing Error (%)": rf4_test_error_percentage,
        "6 Features Training Error (%)": rf6_train_error_percentage,
        "6 Features Testing Error (%)": rf6_test_error_percentage,
    })

    # Save the results to a CSV file
    results_df.to_csv(utilities.results_path + "random_forest_results.csv", index=False)

    print("Finished Random Forest Tests...")
    #print("\nResults Summary:\n", results_df)

if __name__ == "__main__":
    main()
