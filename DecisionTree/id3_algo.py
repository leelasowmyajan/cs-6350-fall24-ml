import math

class Treenode:
    def __init__(self) -> None:
        self.label = None
        self.children = {}
        self.depth = 0

    def add_child(self, value: str, child_node: 'Treenode') -> None:
        self.children[value] = child_node

class ID3_ALGO:
    def __init__(self, label, heuristic='entropy', max_depth=6) -> None:
        self.label = label
        self.heuristic = heuristic  # Possible heuristics = 'entropy', 'gini', or 'majority_error'
        self.max_depth = max_depth 
        self.root = None
    
    def calculate_info_gain(self, data, attribute, attribute_values, label):
        """
        Calculate the information gain based on the selected heuristic.
        """
        # Total impurity before splitting
        label_data = [entry[label] for entry in data]
        total_impurity = self.calculate_impurity(label_data)

        # Weighted impurity on splitting
        weighted_impurity = 0
        for value in attribute_values:
            subset = [entry for entry in data if entry[attribute] == value]
            if subset:
                label_subset = [subset_entry[label] for subset_entry in subset]
                impurity = self.calculate_impurity(label_subset)
                weighted_impurity += (len(subset) / len(data)) * impurity

        # InfoGain = Total impurity - Weighted impurity
        return total_impurity - weighted_impurity

    def calculate_impurity(self, label_data):
        """
        Compute impurity based on the selected heuristic (entropy, gini, or majority error).
        """
        total_data_len = len(label_data)

        # Calculate label distribution 
        label_distribution = {}
        for label in label_data:
            label_distribution[label] = label_distribution.get(label, 0) + 1
        
        # Calculate impurity based on the selected heuristic
        if self.heuristic == 'entropy':
            return self._calculate_entropy(label_distribution, total_data_len)
        elif self.heuristic == 'gini':
            return self._calculate_gini_index(label_distribution, total_data_len)
        elif self.heuristic == 'majority_error':
            return self._calculate_majority_error(label_distribution, total_data_len)

    def _calculate_entropy(self, label_distribution, total_data_len):
        """Calculate entropy."""
        #no need to calculate if set size is 0 or 1
        if total_data_len < 2:
            return 0
        
        entropy = 0
        for count in label_distribution.values():
            prob = count / total_data_len
            entropy -= prob * math.log2(prob)
        return entropy

    def _calculate_gini_index(self, label_distribution, total_data_len):
        """Calculate Gini index."""
        gini = 1
        for count in label_distribution.values():
            prob = count / total_data_len
            gini -= prob ** 2
        return gini

    def _calculate_majority_error(self, label_distribution, total_data_len):
        """Calculate majority error."""
        if total_data_len == 0:
            return 0
        
        min_count = label_distribution[min(label_distribution, key=label_distribution.get)] / total_data_len   
        return min_count if min_count < 1.0 else 0
    
    def build_decision_tree(self, data, attributes, depth=1):
        # Create the current node
        treenode = Treenode()
        treenode.depth = depth

        # Return node if same label
        if len(set([entry[self.label] for entry in data])) == 1:
            treenode.label = data[0][self.label]
            return treenode

        # Return a leaf node if max depth reached or if list is empty
        if depth > self.max_depth or len(attributes) == 0:
            label_distribution = {}
            for entry in data:
                label_distribution[entry[self.label]] = label_distribution.get(entry[self.label], 0) + 1
            common_label = max(label_distribution, key=label_distribution.get)

            treenode.label = common_label
            return treenode

        # Calculate the best attribute to split on
        info_gains = {}
        for attribute, values in attributes.items():
            info_gains[attribute] = self.calculate_info_gain(data, attribute, values, self.label)
        
        best_attribute = max(info_gains, key=info_gains.get)
        treenode.label = best_attribute

        # Split and create child nodes
        for value in attributes[best_attribute]:

            #Calculating subset of the data based on specific attribute and its value
            subset = [entry for entry in data if entry[best_attribute] == value]
            
            if len(subset) == 0:
                child = Treenode()

                label_distribution = {}
                for entry in data:
                    label_distribution[entry[self.label]] = label_distribution.get(entry[self.label], 0) + 1
                common_label = max(label_distribution, key=label_distribution.get)

                child.label = common_label
                #adding child based on the value
                treenode.add_child(value=value, child_node=child)
            else:
                new_attribs = {k: v for k, v in attributes.items() if k != best_attribute}
                #adding child based on the value
                treenode.add_child(value=value, child_node=self.build_decision_tree(subset, new_attribs, depth + 1))  

        return treenode

    def predict(self, treenode, data):
        """Traverse the tree and make a prediction for the data."""
        #if no children means it is leaf node
        if len(treenode.children) == 0:
            return treenode.label
        else:
            attribute = treenode.label
            value = data.get(attribute)
            return self.predict(treenode.children[value], data)

