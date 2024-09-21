import id3_algo, data_read

#Define the maximum depth
MAX_DEPTH = 16

def main():
    data_desc = {
    "labels": ["yes", "no"],
    "columns": [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ],
    "attributes": {
        "age":          ["T", "F"],
        "job":          ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
        "marital":      ["married","divorced","single"],
        "education":    ["unknown","secondary","primary","tertiary"],
        "default":      ["yes","no"],
        "balance":      ["T", "F"],
        "housing":      ["yes","no"],
        "loan":         ["yes","no"],
        "contact":      ["unknown","telephone","cellular"],
        "day":          ["T", "F"],
        "month":        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "duration":     ["T", "F"],
        "campaign":     ["T", "F"],
        "pdays":        ["T", "F"],
        "previous":     ["T", "F"],
        "poutcome":     ["unknown","other","failure","success"]
        }
    }

    train_data = data_read.read_csv("bank/train.csv", data_desc)
    test_data = data_read.read_csv("bank/test.csv", data_desc)
    train_data_unknown_replaced = data_read.read_csv("bank/train.csv", data_desc, replace_unknowns = True)
    test_data_unknown_replaced = data_read.read_csv("bank/test.csv", data_desc, replace_unknowns = True)

    heuristics = ['entropy', 'majority_error', 'gini']
    data_types = ["UNKNOWN NOT REPLACED", "UNKNOWN REPLACED WITH MAX COMMON"]
    error_results = {data_types[0]: {}, data_types[1]: {}}

    for train_data_current, test_data_current, data_type in [
        (train_data, test_data, data_types[0]),
        (train_data_unknown_replaced, test_data_unknown_replaced, data_types[1])
    ]:
        for i in range(1, MAX_DEPTH + 1):
            error_result = { "entropy": {"training": {}, "testing": {}}, 
                            "majority_error": {"training": {}, "testing": {}}, 
                            "gini": {"training": {}, "testing": {}} }

            for heuristic in heuristics:
                # Create and train the decision tree for the current heuristic
                bank_id3 = id3_algo.ID3_ALGO("y", heuristic, i)
                tree_root = bank_id3.build_decision_tree(train_data_current, data_desc["attributes"])

                # Evaluate on the test set
                misclassified_count_test = sum(bank_id3.make_prediction(tree_root, t) != t["y"] for t in test_data_current)
                error_result[heuristic]["testing"]["error_perc"] = round((misclassified_count_test / len(test_data_current)) * 100, 3)

                # Evaluate on the training set
                misclassified_count_train = sum(bank_id3.make_prediction(tree_root, t) != t["y"] for t in train_data_current)
                error_result[heuristic]["training"]["error_perc"] = round((misclassified_count_train / len(train_data_current)) * 100, 3)

            error_results[data_type][i] = error_result

    # Now print all results 
    for data_type, results in error_results.items():
        print(f"\n============== {data_type} ==============")
        for heuristic in heuristics:
            print(f"\n---------- {heuristic} ----------")
            print(f"{'Depth':<10}{'Train Error (%)':<20}{'Test Error (%)':<20}")
            for depth in range(1, MAX_DEPTH + 1):
                train_error = results[depth][heuristic]["training"]["error_perc"]
                test_error = results[depth][heuristic]["testing"]["error_perc"]
                print(f"{depth:<10}{train_error:<20}{test_error:<20}")

if __name__ == "__main__":
    main()