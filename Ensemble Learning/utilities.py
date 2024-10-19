import pandas as pd

#defining paths used in Ensemble Learning folder
bank_train = "bank-1/train.csv" 
bank_test = "bank-1/test.csv"

results_path = "results/" 

#added this here so that i can control the number of iterations for the trees
num_iterations = 500

sample_perc = 0.5

tree_number = 100

sub_attribute_len = 4

debug_mode = False

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