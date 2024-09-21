import statistics

def read_csv(filepath, data_desc, replace_unknowns=False):
    """
    Generalized function to read CSV files for both car and bank datasets.
    Handles 'unknown' values for bank dataset and converts numerical columns to binary if required.
    """
    data = []
    columns_with_unknowns = []  # Tracks columns with 'unknown' values
    columns = data_desc["columns"]
    attributes = data_desc["attributes"]
    numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

    # Read the CSV file and handle 'unknown' values and numerical data if required
    with open(filepath, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            entry = {}
            for i in range(len(terms)):
                if columns[i] in numerical_columns:
                    # convert numerical columns to float for the bank dataset
                    entry[columns[i]] = float(terms[i])
                else:
                    entry[columns[i]] = terms[i]
                    if terms[i] == "unknown" and columns[i] not in columns_with_unknowns:
                        columns_with_unknowns.append(columns[i])
            data.append(entry)

    # numerical binary conversion if required
    if any(attribute == ["T", "F"] for attribute in attributes.values()):
        _convert_numerical_to_binary(data, attributes)

    if replace_unknowns:
        _replace_unknowns(data, columns_with_unknowns, columns)

    return data


def _convert_numerical_to_binary(data, attributes):
    """
    Converts numerical columns to binary ('T' or 'F') based on the median of the dataset.
    """
    attribute_values = ["T", "F"]
    for attribute, possible_values in attributes.items():
        if possible_values == attribute_values:  
            median_value = statistics.median([float(entry[attribute]) for entry in data])
            for entry in data:
                entry[attribute] = "F" if float(entry[attribute]) < median_value else "T"

def _replace_unknowns(data, columns_with_unknowns, columns):
    """
    Replaces 'unknown' values in the dataset with the most common label found in that column.
    """
    for column in columns_with_unknowns:
        label_distribution = {}
        for entry in data:
            if entry[column] != "unknown":
                label_distribution[entry[column]] = label_distribution.get(entry[column], 0) + 1

        # find most common value for column
        most_common_value = max(label_distribution, key=label_distribution.get)

        # replace 'unknown' values with most_common_value
        for entry in data:
            if entry[column] == "unknown":
                entry[column] = most_common_value
