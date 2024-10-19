import numpy as np
import pandas as pd
import utilities

learning_rate=1
tolerance=1e-6

def batch_gd(features, labels, max_iters=1000):
    current_learning_rate = learning_rate
    average_cost_history = []
    learned_weights = None
    has_converged = False
    final_cost_values = []
    iterations = 0

    while not has_converged:
        weights = np.zeros((1, features.shape[1]))
        cost_values = []
        finite_cost_values = []
        average_cost = 0

        for iteration in range(max_iters):
            # Predict the output with current weights
            predictions = features @ weights.T
            error = labels - predictions

            # Compute gradient and update weights
            gradient = -features.T @ error
            next_weights = weights - current_learning_rate * gradient.T

            # Compute and store the cost for the updated weights
            cost = compute_cost(features, labels, next_weights)
            cost_values.append(cost)

            # Check for convergence based on tolerance
            if np.linalg.norm(next_weights - weights) <= tolerance:
                has_converged = True
                learned_weights = next_weights
                final_cost_values = cost_values
                iterations = iteration
                break

            weights = next_weights

        for i in range(len(cost_values)):
            if np.isfinite(cost_values[i]) == False:
                break
            finite_cost_values.append(cost_values[i])
        for v in finite_cost_values:
            average_cost += v / len(finite_cost_values)

        average_cost_history.append((current_learning_rate, average_cost, -1 if np.isfinite(cost_values[i]) else i))
        
        current_learning_rate *= 0.5

    return learned_weights, average_cost_history, final_cost_values, iterations

def compute_cost(features, targets, weights):
    """
    calculating the least mean squares (LMS) cost for given features, targets, and weights.
    """
    errors = targets - features @ weights.T  
    cost = 0.5 * np.sum(np.square(errors))  
    return cost

def main():
    learning_rates = []
    costs = []

    print("Starting BGD...")

    # load training and testing data
    train_features, train_targets = utilities.read_csv(utilities.concrete_train)
    test_features, test_targets = utilities.read_csv(utilities.concrete_test)

    weights, cost_history, final_costs, iterations_count = batch_gd(train_features, train_targets, 10000)

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
    results_df.to_csv(utilities.results_path + "bgd_results.csv", index=False)

    print("Finished BGD...")

if __name__ == "__main__":
    main()

