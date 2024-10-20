from DecisionTree import id3_algo
import pandas as pd
import numpy as np
import utilities

class RandomForest:
    def __init__(self, target_label, sample_size, feature_subset_size, heuristic='entropy', debug_mode=False) -> None:
        self._target_label = target_label
        self._heuristic = heuristic
        self._trees = [] 
        self._sample_size = sample_size 
        self._feature_subset_size = feature_subset_size
        self._debug_mode = debug_mode 

    def train_forest(self, data, features, num_trees=250):
        for _ in range(num_trees):
            self.build_single_tree(data, features)

    def build_single_tree(self, data, features):
        sub_samples = data.sample(n=self._sample_size, replace=True)
        tree = id3_algo.ID3_ALGO_EXTENDED(self._target_label, heuristic=self._heuristic).build_decision_tree_with_subset(sub_samples.to_dict(orient="records"), features, self._feature_subset_size)        
        self._trees.append(tree)

    def predict_with_forest(self, instance):
        vote_sum = 0
        for tree in self._trees:
            vote_sum += int(tree.predict(instance)) 
        return np.sign(vote_sum)
    
    def get_tree(self, index):
        return self._trees[index]

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
    rf2_model = RandomForest("y", sample_size, 2, debug_mode=utilities.debug_mode)
    rf4_model = RandomForest("y", sample_size, 4, debug_mode=utilities.debug_mode)
    rf6_model = RandomForest("y", sample_size, 6, debug_mode=utilities.debug_mode)
    
    # Run the random forest tests for each iteration
    for iteration in iterations:
        print(f"\n=== Iteration {iteration + 1} ===")

        # Train each Random Forest model with the specified number of features
        rf2_model.build_single_tree(train_data, utilities.attributes)
        rf4_model.build_single_tree(train_data, utilities.attributes)
        rf6_model.build_single_tree(train_data, utilities.attributes)

        # Evaluate models on the training set
        tr_rf2_wrong = sum([1 for i in range(num_train_samples) if train_data["y"][i] != rf2_model.predict_with_forest(train_data.iloc[i])])
        tr_rf4_wrong = sum([1 for i in range(num_train_samples) if train_data["y"][i] != rf4_model.predict_with_forest(train_data.iloc[i])])
        tr_rf6_wrong = sum([1 for i in range(num_train_samples) if train_data["y"][i] != rf6_model.predict_with_forest(train_data.iloc[i])])

        rf2_train_error_percentage.append(tr_rf2_wrong / num_train_samples * 100)
        rf2_train_error_total.append(tr_rf2_wrong)

        rf4_train_error_percentage.append(tr_rf4_wrong / num_train_samples * 100)
        rf4_train_error_total.append(tr_rf4_wrong)

        rf6_train_error_percentage.append(tr_rf6_wrong / num_train_samples * 100)
        rf6_train_error_total.append(tr_rf6_wrong)

        # Evaluate models on the testing set
        te_rf2_wrong = sum([1 for i in range(num_test_samples) if test_data["y"][i] != rf2_model.predict_with_forest(test_data.iloc[i])])
        te_rf4_wrong = sum([1 for i in range(num_test_samples) if test_data["y"][i] != rf4_model.predict_with_forest(test_data.iloc[i])])
        te_rf6_wrong = sum([1 for i in range(num_test_samples) if test_data["y"][i] != rf6_model.predict_with_forest(test_data.iloc[i])])

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
