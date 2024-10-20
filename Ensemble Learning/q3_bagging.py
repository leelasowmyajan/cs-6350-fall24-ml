from bagging import Bagging
import numpy as np
import pandas as pd
import utilities

def main():
    print("Running Q3 - Credit Default Dataset Bagging")

    data = utilities.read_xls("default of credit card clients.xls")

    # Handle the continuous features 
    data = utilities.handling_continuous_features(data, utilities.continuous_features)

    # Randomly choose 24000 examples for training and the remaining for testing
    train_data = data.sample(n=24000, random_state=utilities.random_seed_value)
    test_data = data.drop(train_data.index)

    features = [col for col in data.columns if col not in ['Y', 'default payment next month']]

    attributes = {feature: list(train_data[feature].unique()) for feature in features}

    num_train_samples = len(train_data)
    sample_size = int(utilities.sample_perc * num_train_samples)  

    # Initialize Bagging model
    bagging_model = Bagging("Y", sample_size, debug_mode=True)

    # Ensure that all samples have equal weights 
    train_data["weight"] = 1

    # Lists to store errors over iterations
    overall_train_errors = []
    overall_test_errors = []

    # Run Bagging for the specified number of iterations
    for iteration in range(utilities.num_iterations):
        print(f"\n=== Bagging Iteration {iteration + 1} ===")

        # Train the Bagging model
        sub_samples = train_data.sample(n=sample_size, replace=True)
        sub_samples["weight"] = 1  # Assign equal weights to the bootstrapped sample
        bagging_model.train_single_learner(sub_samples, attributes)

        # Training Error
        total_train_errors = sum([1 for i in range(num_train_samples) 
                                  if train_data["Y"].iloc[i] != bagging_model.predict_with_ensemble(train_data.iloc[i])])
        overall_train_errors.append((total_train_errors / num_train_samples) * 100)
        print(f"Training Error: {total_train_errors}/{num_train_samples} ({(total_train_errors / num_train_samples) * 100:.2f}%)")

        # Testing Error
        num_test_samples = len(test_data)
        total_test_errors = sum([1 for i in range(num_test_samples) 
                                 if test_data["Y"].iloc[i] != bagging_model.predict_with_ensemble(test_data.iloc[i])])
        overall_test_errors.append((total_test_errors / num_test_samples) * 100)
        print(f"Testing Error: {total_test_errors}/{num_test_samples} ({(total_test_errors / num_test_samples) * 100:.2f}%)")

    # Store the results in a df
    results_df = pd.DataFrame({
        "Iteration": list(range(utilities.num_iterations)),
        "Training Error (%)": overall_train_errors,
        "Testing Error (%)": overall_test_errors
    })

    # Save the results to a CSV file
    results_df.to_csv(utilities.results_path + "bagging_credit_default_results.csv", index=False)

    print("Finished Bagging on Credit Default Dataset.")

if __name__ == "__main__":
    main()