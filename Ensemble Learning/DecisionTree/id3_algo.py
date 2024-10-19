import math
import random

class Treenode:
    def __init__(self) -> None:
        self.label = None
        self.children = {}
        self.depth = 0

    def add_child(self, value: str, child_node: 'Treenode') -> None:
        self.children[value] = child_node

    def predict(self, data) -> str:
        if not self.children:
            return self.label
        else:
            return self.children[data[self.label]].predict(data)

class ID3_ALGO_EXTENDED:
    def __init__(self, label, heuristic='entropy') -> None:
        self.label = label
        self.heuristic = heuristic  # Possible heuristics = 'entropy', 'gini', or 'majority_error'
        self.root = None
    
    def calculate_info_gain(self, data, attribute, attribute_values, label):
        """
        Calculate the information gain based on the selected heuristic.
        """
        # Total impurity before splitting
        label_data = [entry[label] for entry in data]

        weights = [entry["weight"] for entry in data]

        total_impurity = self.calculate_impurity(label_data, weights)

        # Weighted impurity on splitting
        weighted_impurity = 0
        for value in attribute_values:
            subset = [entry for entry in data if entry[attribute] == value]
            if subset:
                label_subset = [subset_entry[label] for subset_entry in subset]
                s_weights = [subset_entry["weight"] for subset_entry in subset]  
                impurity = self.calculate_impurity(label_subset, s_weights)
                weighted_impurity += (len(subset) / len(data)) * impurity

        # InfoGain = Total impurity - Weighted impurity
        return total_impurity - weighted_impurity

    def calculate_impurity(self, label_data, weights):
        """
        Compute impurity based on the selected heuristic (entropy, gini, or majority error).
        """
        total_data_len = len(label_data)

        # Calculate label distribution 
        label_distribution = {-1: 0, 1: 0}
        
        for i in range(total_data_len):
            label_distribution[label_data[i]] += weights[i]
        
        # Calculate impurity based on the selected heuristic
        if self.heuristic == 'entropy':
            return self._calculate_entropy(label_distribution, total_data_len, weights)
        elif self.heuristic == 'gini':
            return self._calculate_gini_index(label_distribution, total_data_len, weights)
        elif self.heuristic == 'majority_error':
            return self._calculate_majority_error(label_distribution, total_data_len, weights)
    
    def _calculate_entropy(self, label_distribution, total_data_len, weights):
        """Calculate entropy."""
        #no need to calculate if set size is 0 or 1
        if total_data_len < 2:
            return 0
        
        entropy = 0
        total = sum(weights)
        for count in label_distribution.values():
            prob = count / total
            entropy -= prob * (math.log2(prob) if prob > 0.0 else 0)
        return entropy

    def _calculate_gini_index(self, label_distribution, total_data_len, weights):
        """Calculate Gini index."""
        gini = 1
        total = sum(weights)
        for count in label_distribution.values():
            prob = count / total
            gini -= prob ** 2
        return gini

    def _calculate_majority_error(self, label_distribution, total_data_len, weights):
        """Calculate majority error."""
        if total_data_len == 0:
            return 0
        
        total = sum(weights)
        min_count = label_distribution[min(label_distribution, key=label_distribution.get)] / total   
        return min_count if min_count < 1.0 else 0
    
    def get_subset(self, data, attribute, attributes, value) -> list:
        return [entry for entry in data if entry[attribute] == value]
    
    def build_decision_tree(self, data, attributes, depth=0):
        # Create the current node
        treenode = Treenode()
        treenode.depth = depth

        # Return node if same label
        if len(set([entry[self.label] for entry in data])) == 1:
            treenode.label = data[0][self.label]
            return treenode

        # Return a leaf node if max depth reached or if list is empty
        if len(attributes) == 0:
            treenode.label = self.get_most_common_label(data)
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
            #subset = [entry for entry in data if entry[best_attribute] == value]
            subset = self.get_subset(data, best_attribute, attributes, value)
            
            if len(subset) == 0:
                child = Treenode()
                child.label = self.get_most_common_label(data)
                #adding child based on the value
                treenode.add_child(value=value, child_node=child)
            else:
                new_attribs = {k: v for k, v in attributes.items() if k != best_attribute}
                #adding child based on the value
                treenode.add_child(value=value, child_node=self.build_decision_tree(subset, new_attribs, depth + 1))  

        return treenode

    def build_decision_tree_with_subset(self, data, attributes, attrib_count = 2, depth=0):
        # Create the current node
        treenode = Treenode()
        treenode.depth = depth

        # Return node if same label
        if len(set([entry[self.label] for entry in data])) == 1:
            treenode.label = data[0][self.label]
            return treenode

        # Return a leaf node if max depth reached or if list is empty
        if len(attributes) == 0:
            treenode.label = self.get_most_common_label(data)
            return treenode

        # if # of attribs requested is greater than what is available, just take the available amount
        act_attrib_count = attrib_count
        if attrib_count > len(attributes):
            act_attrib_count = len(attributes)

        sub_attribs = {}
        a = list(attributes)

        while len(sub_attribs) < act_attrib_count:
            choice = random.choice(a)
            if choice not in sub_attribs:
                sub_attribs[choice] = attributes[choice]

        # Calculate the best attribute to split on
        info_gains = {}
        for attribute, values in sub_attribs.items():
            info_gains[attribute] = self.calculate_info_gain(data, attribute, values, self.label)
        
        best_attribute = max(info_gains, key=info_gains.get)
        treenode.label = best_attribute

        # Split and create child nodes
        for value in attributes[best_attribute]:

            #Calculating subset of the data based on specific attribute and its value
            #subset = [entry for entry in data if entry[best_attribute] == value]
            subset = self.get_subset(data, best_attribute, attributes, value)
            
            if len(subset) == 0:
                child = Treenode()
                child.label = self.get_most_common_label(data)
                #adding child based on the value
                treenode.add_child(value=value, child_node=child)
            else:
                new_attribs = {k: v for k, v in attributes.items() if k != best_attribute}
                #adding child based on the value
                if len(new_attribs) > 0:
                    treenode.add_child(value=value, child_node=self.build_decision_tree(subset, new_attribs, depth + 1))  

        return treenode

    def build_decision_tree_stump(self, data, attributes):
        treenode = Treenode()
        treenode.depth = 0

        # Calculate the best attribute to split on
        info_gains = {}
        for attribute, values in attributes.items():
            info_gains[attribute] = self.calculate_info_gain(data, attribute, values, self.label)

        best_attribute = max(info_gains, key=info_gains.get)
        treenode.label = best_attribute
        
        # Split and create child nodes
        for value in attributes[best_attribute]:
            #Calculating subset of the data based on specific attribute and its value
            subset = self.get_subset(data, best_attribute, attributes, value)
            child = Treenode()
            child.label = self.get_most_common_label(subset)
            #adding child based on the value
            treenode.add_child(value=value, child_node=child)

        return treenode

    def get_most_common_label(self, data):
        count = {-1:0, 1:0}

        labels = [entry[self.label] for entry in data]
        weights = [entry["weight"] for entry in data]

        for i in range(len(labels)):
            count[labels[i]] += weights[i]
    
        return max(count, key=count.get)

