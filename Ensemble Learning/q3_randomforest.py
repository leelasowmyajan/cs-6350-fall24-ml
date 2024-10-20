from randomforest import RandomForest
import numpy as np
import pandas as pd
import utilities

def main():
    print("Running Q3 - Credit Default Dataset Random Forest")

    # Load the data
    data = utilities.read_xls("default of credit card clients.xls")

    # Handle the continuous features by binning
    data = utilities.handling_continuous_features(data, utilities.continuous_features)

    # Randomly choose 24000 examples for training and the remaining for testing
    train_data = data.sample(n=24000, random_state=utilities.random_seed_value)
    test_data = data.drop(train_data.index)

    # Extract the list of features (excluding the target columns 'Y' and 'default payment next month')
    features = [col for col in data.columns if col not in ['Y', 'default payment next month']]

    # Prepare the attributes dictionary (mapping each feature to its unique values)
    attributes = {feature: list(train_data[feature].unique()) for feature in features}

    # Define the number of training samples and the sample size for bagging
    num_train_samples = len(train_data)
    sample_size = int(utilities.sample_perc * num_train_samples)

    # Assign equal weights to all training samples
    train_data["weight"] = 1

    # Initialize Random Forest models with different numbers of features
    rf2_model = RandomForest("Y", sample_size, 2, debug_mode=True)  # 2 feature subsets
    rf4_model = RandomForest("Y", sample_size, 4, debug_mode=True)  # 4 feature subsets
    rf6_model = RandomForest("Y", sample_size, 6, debug_mode=True)  # 6 feature subsets

    # Initialize lists for storing errors
    rf2_train_error_percentage, rf2_test_error_percentage = [], []
    rf4_train_error_percentage, rf4_test_error_percentage = [], []
    rf6_train_error_percentage, rf6_test_error_percentage = [], []

    # Run Random Forest for the specified number of iterations
    for iteration in range(utilities.num_iterations):
        print(f"\n=== Random Forest Iteration {iteration + 1} ===")

        # Train each Random Forest model
        rf2_model.build_single_tree(train_data, attributes)
        rf4_model.build_single_tree(train_data, attributes)
        rf6_model.build_single_tree(train_data, attributes)

        # Evaluate models on the training set
        rf2_train_error = sum([1 for i in range(num_train_samples) if train_data["Y"].iloc[i] != rf2_model.predict_with_forest(train_data.iloc[i])])
        rf4_train_error = sum([1 for i in range(num_train_samples) if train_data["Y"].iloc[i] != rf4_model.predict_with_forest(train_data.iloc[i])])
        rf6_train_error = sum([1 for i in range(num_train_samples) if train_data["Y"].iloc[i] != rf6_model.predict_with_forest(train_data.iloc[i])])

        rf2_train_error_percentage.append((rf2_train_error / num_train_samples) * 100)
        rf4_train_error_percentage.append((rf4_train_error / num_train_samples) * 100)
        rf6_train_error_percentage.append((rf6_train_error / num_train_samples) * 100)

        # Evaluate models on the testing set
        num_test_samples = len(test_data)
        rf2_test_error = sum([1 for i in range(num_test_samples) if test_data["Y"].iloc[i] != rf2_model.predict_with_forest(test_data.iloc[i])])
        rf4_test_error = sum([1 for i in range(num_test_samples) if test_data["Y"].iloc[i] != rf4_model.predict_with_forest(test_data.iloc[i])])
        rf6_test_error = sum([1 for i in range(num_test_samples) if test_data["Y"].iloc[i] != rf6_model.predict_with_forest(test_data.iloc[i])])

        rf2_test_error_percentage.append((rf2_test_error / num_test_samples) * 100)
        rf4_test_error_percentage.append((rf4_test_error / num_test_samples) * 100)
        rf6_test_error_percentage.append((rf6_test_error / num_test_samples) * 100)

        # Print the errors for this iteration
        print(f"[{iteration + 1} | 2 Features] TRAINING: {rf2_train_error}/{num_train_samples} ({rf2_train_error / num_train_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 4 Features] TRAINING: {rf4_train_error}/{num_train_samples} ({rf4_train_error / num_train_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 6 Features] TRAINING: {rf6_train_error}/{num_train_samples} ({rf6_train_error / num_train_samples * 100:.2f}%)")

        print(f"[{iteration + 1} | 2 Features] TESTING: {rf2_test_error}/{num_test_samples} ({rf2_test_error / num_test_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 4 Features] TESTING: {rf4_test_error}/{num_test_samples} ({rf4_test_error / num_test_samples * 100:.2f}%)")
        print(f"[{iteration + 1} | 6 Features] TESTING: {rf6_test_error}/{num_test_samples} ({rf6_test_error / num_test_samples * 100:.2f}%)")

    # Store the results in a DataFrame
    results_df = pd.DataFrame({
        "Iteration": list(range(utilities.num_iterations)),
        "2 Features Training Error (%)": rf2_train_error_percentage,
        "2 Features Testing Error (%)": rf2_test_error_percentage,
        "4 Features Training Error (%)": rf4_train_error_percentage,
        "4 Features Testing Error (%)": rf4_test_error_percentage,
        "6 Features Training Error (%)": rf6_train_error_percentage,
        "6 Features Testing Error (%)": rf6_test_error_percentage,
    })

    # Save the results to a CSV file
    results_df.to_csv(utilities.results_path + "random_forest_credit_default_results.csv", index=False)

    print("Finished Random Forest on Credit Default Dataset.")

if __name__ == "__main__":
    main()
