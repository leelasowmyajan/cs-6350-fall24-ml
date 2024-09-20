import id3_algo,utilities

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

    train_data = utilities.read_train_csv("car/train.csv", data_desc)
    test_data = utilities.read_test_csv("car/test.csv", data_desc)

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
            misclassified_count = sum(car_id3.make_prediction(tree_root, t) != t["label"] for t in test_data)
            error_result[heuristic]["testing"]["error_perc"] = round((misclassified_count / len(test_data)) * 100, 3)

            # Evaluate on the training set
            misclassified_count = sum(car_id3.make_prediction(tree_root, t) != t["label"] for t in train_data)
            error_result[heuristic]["training"]["error_perc"] = round((misclassified_count / len(train_data)) * 100, 3)
        
        error_results[i] = error_result
    
    for heuristic in heuristics:
        print(f"----------{heuristic}----------")
        for i in range(1, MAX_DEPTH+1):
            print(f"Depth {i} - {error_results[i][heuristic]}")
        print("\n")

if __name__ == "__main__":
    main()