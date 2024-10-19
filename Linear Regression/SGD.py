import numpy as np
import pandas as pd
import utilities
from BGD import compute_cost

learning_rate=1
tolerance=1e-6

def stochastic_gd(features, labels, max_iters=1000):
    current_learning_rate = learning_rate
    average_cost_history = []
    learned_weights = None
    has_converged = False
    final_cost_values = []
    num_samples = features.shape[0]
    iterations = 0
    previous_cost = float('inf') # starting w a very high cost

    np.random.seed(utilities.random_seed_value)

    while not has_converged:
        weights = np.zeros((1, features.shape[1]))
        cost_values = []
        finite_cost_values = []
        encountered_infinity = False

        for iteration in range(max_iters):
            random_index = np.random.randint(num_samples)
            selected_feature = features[random_index]
            selected_label = labels[random_index]

            # calculating gradient
            predicted_value = selected_feature * np.transpose(weights)
            error_value = selected_label - predicted_value

            # vectorized weight update for all features
            weights += current_learning_rate * error_value * selected_feature

            cost = compute_cost(features, labels, weights)
            cost_values.append(cost)

            # checking for convergence based on tolerance
            if np.abs(cost - previous_cost) <= tolerance:
                has_converged = True
                learned_weights, final_cost_values, iterations = weights, cost_values, iteration
                break

            previous_cost = cost
        
        # Filter finite values and finding the infinity
        finite_cost_values = [cost for cost in cost_values if np.isfinite(cost)]
        encountered_infinity = len(finite_cost_values) != len(cost_values)

        # calculate average cost
        avg_cost = sum(finite_cost_values) / len(finite_cost_values) if finite_cost_values else 0

        # adding the learning rate, average cost, and check for infinity
        average_cost_history.append((current_learning_rate, avg_cost, -1 if not encountered_infinity else iteration))

        current_learning_rate *= 0.5

    return learned_weights, average_cost_history, final_cost_values, iterations

def main():
    learning_rates = []
    costs = []

    print("Starting SGD...")

    # load training and testing data
    train_features, train_targets = utilities.read_csv(utilities.concrete_train)
    test_features, test_targets = utilities.read_csv(utilities.concrete_test)
    
    weights, cost_history, final_costs, iterations_count = stochastic_gd(train_features, train_targets, 10000)
    
    print(f"Final weights:\n{weights}")
    print(f"Final learning rate: {cost_history[-1][0]}")
    print(f"Cost of Test Data: {compute_cost(test_features, test_targets, weights)}")

    # displaying cost history to verify convergence
    print("Cost History (Learning Rate | Average Cost | Iterations until divergence if applicable):")

    for learning_rate, average_cost, divergence_iter in cost_history:
        if divergence_iter != -1:
            print(f"Learning Rate: {learning_rate}, Average Cost: {average_cost} (Diverged at iteration {divergence_iter})")
        else:
            print(f"Learning Rate: {learning_rate}, Average Cost: {average_cost}")
        learning_rates.append(learning_rate)
        costs.append(average_cost)

    # saving results to CSV
    iterations = list(range(iterations_count + 1))
    results_df = pd.DataFrame({"Iterations": iterations, "Cost": final_costs})
    results_df.to_csv(utilities.results_path + "sgd_results.csv", index=False)

    print("Finished SGD...")

if __name__ == "__main__":
    main()