def read_train_csv(filepath, data_desc):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            entry = {}
            for i in range(len(terms)):
                entry[data_desc["columns"][i]] = terms[i]
            data.append(entry)
    return data

def read_test_csv(filepath, data_desc):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            entry = {}
            for i in range(len(terms)):
                entry[data_desc["columns"][i]] = terms[i]
            data.append(entry)
    return data