import numpy as np
import copy
import utilities

import warnings
warnings.filterwarnings("ignore")

class SigmoidAct():
    def __init__(self) -> None:
        pass

    def calc_derivative(self, previous_derivative, current_layer_output, next_layer_output):
        return previous_derivative * next_layer_output * current_layer_output * (1 - current_layer_output)

    def calc_output(self, val):
        return 1 / (1 + np.exp(-val))

    def calc_weight_grads(self, weight_grads, previous_derivative, current_layer_output, next_layer_output):
    #calculates gradients of the weights using backpropagation.    
        for i, grads in enumerate(weight_grads):  # iterate over each row in weight_grads
            weight_grads[i] = [
                self.calc_derivative(previous_derivative[j], current_layer_output[j], next_layer_output[i])
                for j in range(1, len(grads) + 1)  # skip bias (index 0)
            ]
        return weight_grads
    
    def calc_first_node_derivatives(self, weights, previous_derivatives, current_layer_outputs):
        #calculates the derivatives for the first node in the current layer.
        
        # initializing node derivatives with zeros matching the shape of current layer outputs
        node_derivatives = np.zeros_like(current_layer_outputs)

        for i, weight_row in enumerate(weights):
            # calculating the weighted sum of previous derivatives for the current node
            node_derivatives[i] = sum(weight * previous_derivatives[j] for j, weight in enumerate(weight_row))

        return node_derivatives

    def calc_node_derivatives(self, weights, previous_derivatives, current_layer_outputs):
        # initializing node derivatives with zeros matching the shape of current layer outputs
        node_derivatives = np.zeros_like(current_layer_outputs)

        # iterating over nodes in the current layer
        for i, weight_row in enumerate(weights):
            # calculating the weighted sum of previous derivatives for the current node
            node_derivatives[i] = sum(weight * previous_derivatives[j + 1] for j, weight in enumerate(weight_row))

        return node_derivatives
    
    def calc_activations(self, vals, weights, layer_size):
    #calculates the activations for all nodes in the current layer.
        # initializing layer outputs with zeros and setting the bias node to 1
        layer_outputs = np.zeros(layer_size)
        layer_outputs[0] = 1  # setting bias node

        # iterating over the nodes in the current layer (excluding the bias node)
        for i in range(1, layer_size):
            # calculating the weighted sum of inputs for the current node
            layer_outputs[i] = self.calc_output(sum(input_val * weights[j][i - 1] for j, input_val in enumerate(vals)))

        return layer_outputs

class LinearAct():
    def __init__(self) -> None:
        pass

    def calc_derivative(self, previous_derivative, current_layer_output, next_layer_output):
        return previous_derivative[0] * next_layer_output

    def calc_activations(self, vals, weights, layer_size):
        # initializing an array to store the outputs of the current layer
        layer_outputs = np.zeros(layer_size)

        # iterating over each node in the current layer
        for i in range(layer_size):
            # calculating the weighted sum of inputs and weights for the current node
            layer_outputs[i] = sum(vals[j] * weights[j][i] for j in range(len(vals)))

        return layer_outputs

    def calc_node_derivatives(self, weights, previous_derivatives, current_layer_outputs):
        """
        calculates the derivatives for the nodes in the current layer.
        """
        # initializing an array to store node derivatives
        node_derivatives = np.zeros(current_layer_outputs.shape)

        # iterating over nodes in the current layer
        for i in range(len(node_derivatives)):
            # calculating the weighted sum of previous derivatives and weights
            node_derivatives[i] = sum(weights[i][j] * previous_derivatives[j + 1] for j in range(len(weights[i])))

        return node_derivatives
    
    def calc_first_node_derivatives(self, weights, previous_derivatives, current_layer_outputs):
        """
        calculates the derivatives for the first node in the current layer.
        """
        # initializing an array to store node derivatives
        node_derivatives = np.zeros(current_layer_outputs.shape)

        # iterating over nodes in the current layer
        for i in range(len(node_derivatives)):
            # calculating the weighted sum of previous derivatives and weights
            node_derivatives[i] = sum(weights[i][j] * previous_derivatives[j] for j in range(len(weights[i])))

        return node_derivatives

    def calc_weight_grads(self, weight_grads, previous_derivative, current_layer_output, next_layer_output, activation_func):
        """
        calculates gradients of the weights using backpropagation.
        """
        # iterating over each row of weight gradients
        for i in range(len(weight_grads)):
            # iterating over each element in the row
            for j in range(len(weight_grads[i])):
                # calculating and updating the gradient value
                weight_grads[i][j] = activation_func.calc_derivative(
                    previous_derivative[j], 
                    current_layer_output[j], 
                    next_layer_output[i]
                )
        return weight_grads

class NeuralNetwork():
    def __init__(self, layers, weights, layer_widths):
        self.layers = layers
        self.weights = weights
        self.layer_widths = layer_widths
        self.layer_outputs = None
    
    def predict_batch(self, vals):
        #predicts the output for a batch of input values.
        return [self.predict_single(val) for val in vals]

    def predict_single(self, val):
        return np.sign(self.forward_pass(val))

    def train_network(self, vals, labels, decay_rate, learning_rate, T=100):
        # generating shuffled indices for the training data
        shuffled_indices = generate_shuffled_indices(vals, T)
        
        # iterating over the number of epochs
        for t in range(T):
            # calculating the current learning rate based on the decay schedule
            current_learning_rate = self.lr_sched(learning_rate, decay_rate, t)
            
            for idx in shuffled_indices[t]:
                # retrieving the current data point and its label
                val, label = vals[idx], labels[idx]
                
                # performing forward propagation
                output = self.forward_pass(val)
                
                # calculating the weight gradients using backward propagation
                weight_grads = self.backward_pass(output - label)
                
                # updating the weights using the calculated gradients
                self.apply_weight_updates(current_learning_rate, weight_grads)

    def apply_weight_updates(self, learning_rate, grads):
        """
        updates the weights of the neural network using the calculated gradients.
        """
        # iterating over each layer of the neural network
        for layer_index, layer_grads in enumerate(grads):
            # iterating over each node in the current layer
            for node_index, node_grads in enumerate(layer_grads):
                # updating weights for each connection of the current node
                self.weights[layer_index][node_index] -= learning_rate * np.array(node_grads)

    def lr_sched(self, intial_lr, lr_decay, t):
        return intial_lr / (1 + intial_lr / lr_decay * t)

    def forward_pass(self, val):
        self.layer_outputs = np.array([np.zeros(w) for w in self.layer_widths], dtype=object)
        self.layer_outputs[0] = val
        for layer_index in range(1, len(self.layer_widths)):
            self.layer_outputs[layer_index] = self.layers[layer_index-1].calc_activations(
                self.layer_outputs[layer_index-1], 
                self.weights[layer_index-1], 
                self.layer_widths[layer_index])
        return self.layer_outputs[-1]

    def backward_pass(self, loss, printOp=False):
        node_derivatives = [loss]
        weight_grads = copy.deepcopy(self.weights)

        last_layer_index = len(self.weights) - 1

        weight_grads[last_layer_index] = self.layers[last_layer_index].calc_weight_grads(
            weight_grads[last_layer_index], 
            node_derivatives, 
            self.layer_outputs[last_layer_index+1], 
            self.layer_outputs[last_layer_index], 
            self.layers[last_layer_index])
        
        node_derivatives = self.layers[last_layer_index].calc_first_node_derivatives(
            self.weights[last_layer_index], 
            node_derivatives, 
            self.layer_outputs[last_layer_index])
        
        if printOp:
            print(f"=== Node Derivatives at Output Layer ===\n{node_derivatives}\n")

        for layer_index in range(last_layer_index - 1, 0, -1):
            weight_grads[layer_index] = self.layers[layer_index].calc_weight_grads(
                weight_grads[layer_index], 
                node_derivatives, 
                self.layer_outputs[layer_index+1], 
                self.layer_outputs[layer_index])
            node_derivatives = self.layers[last_layer_index].calc_node_derivatives(
                self.weights[layer_index], 
                node_derivatives, 
                self.layer_outputs[layer_index])
            if printOp:
                print(f"=== Node Derivatives at Hidden Layer ===\n{node_derivatives}\n")
                
        weight_grads[0] = self.layers[0].calc_weight_grads(
            weight_grads[0], 
            node_derivatives, 
            self.layer_outputs[1], 
            self.layer_outputs[0])

        return weight_grads

def generate_shuffled_indices(data, T):
    """
    Generates shuffled indices for each epoch of training.
    """
    shuffled_indices = []
    num_samples = data.shape[0]
    for _ in range(T):
        indices = np.arange(num_samples)  # generating sequential indices
        np.random.default_rng().shuffle(indices)  # shuffling them randomly
        shuffled_indices.append(indices)
    return shuffled_indices

def validate_using_3Q():
    # defining input example
    input_vector = np.array([1, 1, 1])

    # defining network structure and hardcoded weights
    layer_widths = np.array([3, 3, 3, 1])
    hardcoded_weights = np.array([
        [[-1, 1], [-2, 2], [-3, 3]], 
        [[-1, 1], [-2, 2], [-3, 3]], 
        [[-1], [2], [-1.5]]
    ], dtype=object)

    # creating a neural network instance
    nn_model = NeuralNetwork(utilities.NN_LAYERS, hardcoded_weights, layer_widths)

    # performing forward propagation
    print("=== Forward Propagation ===")
    output = nn_model.forward_pass(input_vector)
    layer_results = nn_model.layer_outputs

    # displaying activations for each layer
    for layer_index in range(len(layer_results)):
        print(f"Layer {layer_index}: {layer_results[layer_index]}")
    print("Neural Network Classification Result:", np.sign(output))

    # performing backward propagation
    print("\n=== Backward Propagation ===")
    gradients = nn_model.backward_pass(output - 1, printOp=True)

    # displaying computed gradients
    print("Computed Gradients:", gradients)

def initialize_random_weights(input_size, output_size):
    weights = []
    # iterating over input size to create rows of weights
    for _ in range(input_size):
        random_weights = np.random.normal(0, 0.1, output_size - 1 if output_size - 1 > 0 else 1).tolist()
        weights.append(random_weights)
    return weights

def initialize_zero_weights(input_size, output_size):
    weights = []
    # iterating over input size to create rows of weights
    for _ in range(input_size):
        weights.append([0] * ((output_size - 1) if output_size - 1 > 0 else 1))
    return weights

def evaluate_weight_initialization(train_features, train_labels, test_features, test_labels, method="random"):
    """
    method = random/zero
    evaluates a neural network's performance with random/zero weight initialization.
    """
    # initializing dictionaries to store training and testing errors
    training_errors = {}
    testing_errors = {}

    def initialize_weights(input_size, output_size, method):
        if method == "random":
            return initialize_random_weights(input_size, output_size)
        elif method == "zero":
            return initialize_zero_weights(input_size, output_size)

    # iterating over predefined hidden layer sizes
    for hidden_layer_size in utilities.HIDDEN_LAYER_SIZES:
        # defining network layer widths (input, two hidden layers, output)
        layer_widths = [len(train_features[0]), hidden_layer_size, hidden_layer_size, 1]

        # initializing weights with random values from a Gaussian distribution
        weights = [
            initialize_weights(layer_widths[layer_index], layer_widths[layer_index + 1], method)
            for layer_index in range(len(layer_widths) - 1)
        ]
        weights = np.array(weights, dtype=object)

        # creating a neural network instance
        nn_model = NeuralNetwork(utilities.NN_LAYERS, weights, layer_widths)

        # training the neural network
        nn_model.train_network(train_features, train_labels, decay_rate=0.1, learning_rate=0.1)

        # evaluating training error
        training_predictions = nn_model.predict_batch(train_features)
        training_errors[hidden_layer_size] = sum(
            int(pred != actual) for pred, actual in zip(training_predictions, train_labels)
        )

        # evaluating testing error
        testing_predictions = nn_model.predict_batch(test_features)
        testing_errors[hidden_layer_size] = sum(
            int(pred != actual) for pred, actual in zip(testing_predictions, test_labels)
        )

        print(f"Finished evaluating NN with hidden layer size {hidden_layer_size}.")

    # displaying errors for all hidden layer sizes
    print(f"\n{method.capitalize()} Initialization Results:")
    for hidden_layer_size in utilities.HIDDEN_LAYER_SIZES:
        training_error_percent = (training_errors[hidden_layer_size] / len(train_labels)) * 100
        testing_error_percent = (testing_errors[hidden_layer_size] / len(test_labels)) * 100
        print(f"Hidden Layer Size: {hidden_layer_size}, Training Error: {float(training_error_percent):.4f}%, Testing Error: {float(testing_error_percent):.4f}%")

def main():
    print("========== Neural Network Implementation ==========\n")

    # backpropagation check
    print("========== BACKPROPAGATION CHECK ==========")
    print("Testing the back-propagation algorithm with Problem 3 example...")
    validate_using_3Q()
    print("Backpropagation test completed successfully.\n")
    
    # loading dataset
    train_features, train_labels = utilities.read_csv(utilities.bank_note_train)
    test_features, test_labels = utilities.read_csv(utilities.bank_note_test)

    # random initialization check
    print("========== RANDOM INITIALIZATION CHECK ==========")
    print("Training the neural network with random weight initialization...\n")
    evaluate_weight_initialization(train_features, train_labels, test_features, test_labels, method="random")
    print("\nRandom initialization training and testing completed.\n")

    # zero initialization check
    print("========== ZERO INITIALIZATION CHECK ==========")
    print("Training the neural network with zero weight initialization...\n")
    evaluate_weight_initialization(train_features, train_labels, test_features, test_labels, method="zero")
    print("\nZero initialization training and testing completed.\n")
    
    print("========== Neural Network Implementation Completed ==========")
    return 0

if __name__ == "__main__":
    main()

