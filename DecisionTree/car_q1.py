import id3_algo,data_read

#Define the maximum depth
MAX_DEPTH = 6

def main():

    data_desc = {
    "labels": ["unacc", "acc", "good", "vgood"],
    "columns": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"],
    "attributes": { "buying":   ["vhigh", "high", "med", "low"],
             "maint":    ["vhigh", "high", "med", "low"],
             "doors":    ["2", "3", "4", "5more"],
             "persons":  ["2", "4", "more"],
             "lug_boot": ["small", "med", "big"],
             "safety":   ["low", "med", "high"]
        }
    }

    train_data = data_read.read_csv("car/train.csv", data_desc)
    test_data = data_read.read_csv("car/test.csv", data_desc)

    heuristics = ['entropy', 'majority_error', 'gini']
    error_results = {}

    for i in range(1, MAX_DEPTH+1):
        error_result = { "entropy": {"training": {}, "testing": {}}, 
                "majority_error": {"training": {}, "testing": {}}, 
                "gini": {"training": {}, "testing": {}} }

        for heuristic in heuristics:
            # Create and train the decision tree for the current heuristic
            car_id3 = id3_algo.ID3_ALGO("label", heuristic, i)
            tree_root = car_id3.build_decision_tree(train_data, data_desc["attributes"])

            # Evaluate on the test set
            misclassified_count_test = sum(car_id3.make_prediction(tree_root, t) != t["label"] for t in test_data)
            error_result[heuristic]["testing"]["error_perc"] = round((misclassified_count_test / len(test_data)) * 100, 3)

            # Evaluate on the training set
            misclassified_count_train = sum(car_id3.make_prediction(tree_root, t) != t["label"] for t in train_data)
            error_result[heuristic]["training"]["error_perc"] = round((misclassified_count_train / len(train_data)) * 100, 3)
        
        error_results[i] = error_result

    # Now print all results 
    for heuristic in heuristics:
        print(f"\n---------- {heuristic} ----------")
        print(f"{'Depth':<10}{'Train Error (%)':<20}{'Test Error (%)':<20}")
        for depth in range(1, MAX_DEPTH + 1):
            train_error = error_results[depth][heuristic]["training"]["error_perc"]
            test_error = error_results[depth][heuristic]["testing"]["error_perc"]
            print(f"{depth:<10}{train_error:<20}{test_error:<20}")

if __name__ == "__main__":
    main()