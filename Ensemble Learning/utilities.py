import pandas as pd

#defining paths used in Ensemble Learning folder
bank_train = "bank-1/train.csv" 
bank_test = "bank-1/test.csv"

results_path = "results/" 
results_adaboost_csv = "results/adaboost_overall_results.csv" 
results_stump_csv = "results/adaboost_stump_results.csv" 
results_bagging_csv = "results/bagging_results.csv"
results_forest_csv = "results/random_forest_results.csv"
results_adaboost_credit_csv = "results/adaboost_credit_default_results.csv"
results_bagging_credit_csv = "results/bagging_credit_default_results.csv"
results_forest_credit_csv = "results/random_forest_credit_default_results.csv"

#added this here so that i can control the number of iterations for the trees
num_iterations = 500
tree_number = 100
sample_perc = 0.5
sub_attribute_len = 4
debug_mode = False
random_seed_value = 16

columns = [ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y" ]
data_types = { "age":int, "job":str, "marital":str, "education":str, "default":str, "balance":int, "housing":str, "loan":str, "contact":str, "day":int, "month":str, "duration":int, "campaign":int, "pdays":int, "previous":int, "poutcome":str, "y":str }
attributes = {
    "age":          [-1, 1],
    "job":          ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
    "marital":      ["married","divorced","single"],
    "education":    ["unknown","secondary","primary","tertiary"],
    "default":      ["yes","no"],
    "balance":      [-1, 1],
    "housing":      ["yes","no"],
    "loan":         ["yes","no"],
    "contact":      ["unknown","telephone","cellular"],
    "day":          [-1, 1],
    "month":        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration":     [-1, 1],
    "campaign":     [-1, 1],
    "pdays":        [-1, 1],
    "previous":     [-1, 1],
    "poutcome":     ["unknown","other","failure","success"]
}
continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
                           'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                           'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

def read_csv(filepath):
    data = pd.read_csv(filepath, names=columns, dtype=data_types)

    # Handling numeric data
    for key in data_types:
        if data_types[key] == int:
            median = data[key].median()
            data[key] = data[key].apply(lambda val: -1 if val < median else 1)
    
    # Converting target
    data["y"] = data["y"].apply(lambda val: -1 if val=="no" else 1)

    return data

def read_xls(filepath):
    data = pd.read_excel(filepath, header=1, engine='xlrd') 

    # Convert the 'default payment next month' column to the target variable 'Y'
    target_column = 'default payment next month'  
    data['Y'] = data[target_column].apply(lambda x: 1 if x == 1 else -1)

    # dropping the 'ID' column as it's unnecessary
    data = data.drop(columns=['ID'])

    return data

def handling_continuous_features(data, continuous_features):
    for feature in continuous_features:
        # Discretize continuous features into 3 bins
        data[feature] = pd.qcut(data[feature], q=3, labels=False)
    return data
