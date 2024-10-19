import numpy as np
import bagging
import utilities

def main():
    print("Starting Q2-C tests...")

    train_data = utilities.read_csv(utilities.bank_train)
    
    # Assign uniform weight to all samples
    train_data["weight"] = 1 

    test_data = utilities.read_csv(utilities.bank_test)
    num_test_samples = len(test_data.index)

    bagging_trees = []
    single_trees = []
    
    # Train all the trees
    print("Building trees...")
    for iteration in range(utilities.tree_number):
        # Sample 1,000 examples uniformly without replacement from the training dataset
        training_subset = train_data.sample(n=1000)
        sample_count = int(utilities.sample_perc * len(training_subset)) # Subsample count

        # Train a bagged model based on the 1,000 training examples and learn trees
        bag_model = bagging.Bagging("y", sample_count)
        bag_model.bag(training_subset, utilities.attributes, utilities.num_iterations)
        bagging_trees.append(bag_model)

        # For comparison, pick the first tree in each run to get individual trees
        single_trees.append(bag_model.get_hypothesis(0))

    print("Finished building trees...")

    print("Generating predictions...")
    single_tree_predictions = []
    bagged_tree_predictions = [] 
    for i in range(num_test_samples):
        # Single tree predictions
        single_predictions = [single_trees[t].predict(test_data.iloc[i]) for t in range(utilities.tree_number)]
        single_tree_predictions.append(single_predictions)

        # Bagged tree predictions
        bagged_predictions = [bagging_trees[t].classify(test_data.iloc[i]) for t in range(utilities.tree_number)]
        bagged_tree_predictions.append(bagged_predictions)

    print("Finished generating predictions...")

    # Calculate biases for both single trees and bagged trees
    single_tree_biases = []
    bagged_tree_biases = []
    single_tree_mean = []
    bagged_tree_mean = []

    print("Calculating bias...")
    for i in range(num_test_samples):
        true_value = test_data["y"][i]

        # Bias for single trees
        single_mean_prediction = np.mean(single_tree_predictions[i])
        single_tree_mean.append(single_mean_prediction)
        single_tree_bias = (single_mean_prediction - true_value) ** 2
        single_tree_biases.append(single_tree_bias)

        # Bias for bagged trees
        bagged_mean_prediction = np.mean(bagged_tree_predictions[i])
        bagged_tree_mean.append(bagged_mean_prediction)
        bagged_tree_bias = (bagged_mean_prediction - true_value) ** 2
        bagged_tree_biases.append(bagged_tree_bias)

    print("Finished calculating bias...")

    # Calculate variances for both single trees and bagged trees
    single_tree_variances = []
    bagged_tree_variances = []
    variance_factor = 1 / (num_test_samples - 1)

    print("Calculating variance...")
    for i in range(num_test_samples):
        # Variance for single trees
        single_variance = variance_factor * sum((x - single_tree_mean[i]) ** 2 for x in single_tree_predictions[i])
        single_tree_variances.append(single_variance)

        # Variance for bagged trees
        bagged_variance = variance_factor * sum((x - bagged_tree_mean[i]) ** 2 for x in bagged_tree_predictions[i])
        bagged_tree_variances.append(bagged_variance)

    print("Finished calculating variance...")

    # Calculate average bias, variance, and general squared error
    avg_single_tree_bias = np.mean(single_tree_biases)
    avg_single_tree_variance = np.mean(single_tree_variances)
    avg_bagged_tree_bias = np.mean(bagged_tree_biases)
    avg_bagged_tree_variance = np.mean(bagged_tree_variances)

    print("Calculating general squared error...")
    single_tree_general_error = [single_tree_biases[i] + single_tree_variances[i] for i in range(num_test_samples)]
    bagged_tree_general_error = [bagged_tree_biases[i] + bagged_tree_variances[i] for i in range(num_test_samples)]

    avg_single_tree_general_error = np.mean(single_tree_general_error)
    avg_bagged_tree_general_error = np.mean(bagged_tree_general_error)

    print("Finished calculating general squared error...")

    print("Average Bias (Single Tree):", avg_single_tree_bias)
    print("Average Variance (Single Tree):",avg_single_tree_variance)
    print("Average General Squared Error (Single Tree):", avg_single_tree_general_error)
    print("Average Bias (Bagged):",avg_bagged_tree_bias)
    print("Average Variance (Bagged):",avg_bagged_tree_variance)
    print("Average General Squared Error (Bagged):", avg_bagged_tree_general_error)

    # Save the output to a text file
    output_file_path = utilities.results_path + "q2c_result.txt"
    with open(output_file_path, "w") as f:
        f.write(f"Average Bias (Single Tree): {avg_single_tree_bias}\n")
        f.write(f"Average Variance (Single Tree): {avg_single_tree_variance}\n")
        f.write(f"Average General Squared Error (Single Tree): {avg_single_tree_general_error}\n\n")
        f.write(f"Average Bias (Bagged Trees): {avg_bagged_tree_bias}\n")
        f.write(f"Average Variance (Bagged Trees): {avg_bagged_tree_variance}\n")
        f.write(f"Average General Squared Error (Bagged Trees): {avg_bagged_tree_general_error}\n")

    print("Results saved to", output_file_path)

    print("Finished Q2-C tests...")
    return 0

if __name__ == "__main__":
    main()