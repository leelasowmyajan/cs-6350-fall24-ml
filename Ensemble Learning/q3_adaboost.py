from adaboost import AdaBoost
import numpy as np
import pandas as pd
import utilities

def main():
    print("Running Q3 - Credit Default Dataset")
    
    # loading the data
    data = utilities.read_xls("default of credit card clients.xls")

    # handling the continuous features
    data = utilities.handling_continuous_features(data, utilities.continuous_features)

    # randomly choose 24000 examples for training and the remaining for testing
    train_data = data.sample(n=24000, random_state=utilities.random_seed_value)
    test_data = data.drop(train_data.index)

    # extracting the list of features 
    features = [col for col in data.columns if col not in ['Y', 'default payment next month']]

    attributes = {feature: list(train_data[feature].unique()) for feature in features}

    num_train_samples = len(train_data)
    train_data["weight"] = 1 / num_train_samples
    actual_train_labels = train_data["Y"].tolist()

    num_test_samples = len(test_data)

    # Initialize AdaBoost model
    adaboost_model = AdaBoost("Y", num_train_samples, debug_mode=True)

    # Lists to store errors over iterations
    overall_train_errors = []
    overall_test_errors = []

    for iteration in range(utilities.num_iterations):
        print(f"\n=== Boosting Round {iteration + 1} ===")

        # Train the AdaBoost model
        adaboost_model.train_single_learner(train_data, attributes, actual_train_labels)

        # Training Error
        overall_train_error = sum([1 for i in range(num_train_samples) 
                                   if train_data["Y"].iloc[i] != adaboost_model.predict_with_ensemble(train_data.iloc[i])])
        overall_train_errors.append((overall_train_error / num_train_samples) * 100)
        print(f"Training Error: {overall_train_error}/{num_train_samples} ({(overall_train_error / num_train_samples) * 100:.2f}%)")

        # Testing Error
        overall_test_error = sum([1 for i in range(num_test_samples) 
                                  if test_data["Y"].iloc[i] != adaboost_model.predict_with_ensemble(test_data.iloc[i])])
        overall_test_errors.append((overall_test_error / num_test_samples) * 100)
        print(f"Testing Error: {overall_test_error}/{num_test_samples} ({(overall_test_error / num_test_samples) * 100:.2f}%)")

    # storing results in df
    results_df = pd.DataFrame({
        "Iteration": list(range(utilities.num_iterations)),
        "Training Error (%)": overall_train_errors,
        "Testing Error (%)": overall_test_errors
    })

    # Save the results to a CSV file
    results_df.to_csv(utilities.results_path + "adaboost_credit_default_results.csv", index=False)

    print("Finished AdaBoost on Credit Default Dataset.")

if __name__ == "__main__":
    main()
