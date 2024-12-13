import torch
import numpy as np
import utilities

import warnings
warnings.filterwarnings("ignore")

class NeuralNetwork(torch.nn.Module):
    def __init__(self, network_depth, layer_width, activation_mode=0) -> None:
        super(NeuralNetwork, self).__init__()

        # setting activation function and weight initialization method
        if activation_mode == 0:
            # tanh / He
            self.activation_function = torch.nn.Tanh()
            self.trainingMethod = self._initialize_weights_he
        elif activation_mode == 1:
            # ReLU / Xavier
            self.activation_function = torch.nn.ReLU()
            self.trainingMethod = self._initialize_weights_xavier

        # defining input layer
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(4, layer_width), self.activation_function
        )

        # defining hidden layers
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(layer_width, layer_width), self.activation_function)
            for _ in range(network_depth - 2)
        ])

        # defining output layer
        self.output_layer = torch.nn.Linear(layer_width, 1)

    def _initialize_weights_xavier(self, layer):
        if isinstance(layer, torch.nn.Linear):  # checking if the layer is a linear layer
            torch.nn.init.xavier_normal_(layer.weight)  # initializing weights with xavier normal
            layer.bias.data.fill_(0.01)  # setting bias values to a small constant

    def _initialize_weights_he(self, layer):
        if isinstance(layer, torch.nn.Linear):  # checking if the layer is a linear layer
            torch.nn.init.kaiming_normal_(layer.weight)  # initializing weights with he normal
            layer.bias.data.fill_(0.01)  # setting bias values to a small constant

    def forward(self, input_data):
        # passing data through input layer
        x = self.input_layer(input_data)

        # passing data through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # passing data through output layer
        return self.output_layer(x)

def train_neural_network(data_loader, nn_model, loss_function, optimizer):
    """
    training the neural network for one epoch.
    returns: list of loss values recorded during training
    """
    nn_model.train()  # setting the model to training mode
    epoch_loss_list = []  # initializing list to store loss values

    for batch_index, (features, labels) in enumerate(data_loader):
        # transferring data to the compute device (cpu/gpu)
        features = features.to(utilities.COMPUTE_DEVICE)
        labels = labels.to(utilities.COMPUTE_DEVICE)

        # forward pass: predicting outputs
        predictions = nn_model(features)
        loss_value = loss_function(torch.reshape(predictions, labels.shape), labels)

        # backpropagation
        optimizer.zero_grad()  # clearing previous gradients
        loss_value.backward()  # calculating gradients
        optimizer.step()  # updating model parameters

        if batch_index % utilities.TRAINING_BATCH_SIZE == 0:
            epoch_loss_list.append(loss_value.item())

    return epoch_loss_list

def evaluate_neural_network(data_loader, nn_model, loss_function):
    """
    evaluating the neural network on the provided dataset
    returns: average loss across all batches
    """
    total_loss = 0  # initializing total loss
    num_batches = len(data_loader)  # number of batches in the data loader
    nn_model.eval()  # setting the model to evaluation mode

    with torch.no_grad():  # disabling gradient calculations
        for features, labels in data_loader:
            # transferring data to the compute device (cpu/gpu)
            features = features.to(utilities.COMPUTE_DEVICE)
            labels = labels.to(utilities.COMPUTE_DEVICE)

            # forward pass: predicting outputs
            predictions = nn_model(features)

            # calculating loss for the current batch
            total_loss += loss_function(torch.reshape(predictions, labels.shape), labels).item()

    return total_loss / num_batches

def main():
    # creating data loaders for training and testing
    train_data_loader = utilities.create_data_loader(
        utilities.bank_note_train, utilities.TRAINING_BATCH_SIZE, shuffle=True
    )
    test_data_loader = utilities.create_data_loader(
        utilities.bank_note_test, utilities.TRAINING_BATCH_SIZE, shuffle=False
    )

    # calculating total combinations of width, depth, and activation mode
    total_combinations = (
        len(utilities.HIDDEN_LAYER_SIZES) 
        * len(utilities.NETWORK_DEPTHS) 
        * len(utilities.TRAINING_MODES)
    )

    print("Starting neural network training with various configurations...")

    # initializing results dictionary to store training and test errors
    experiment_results = {}

    # iterating over hidden layer sizes (widths)
    for layer_width in utilities.HIDDEN_LAYER_SIZES:
        # iterating over network depths
        for network_depth in utilities.NETWORK_DEPTHS:
            # iterating over activation modes (0 for tanh, 1 for ReLU)
            for activation_mode in utilities.TRAINING_MODES:
                configuration_key = (layer_width, network_depth, activation_mode)

                # initializing the neural network with current configuration
                nn_model = NeuralNetwork(
                    network_depth=network_depth, 
                    layer_width=layer_width, 
                    activation_mode=activation_mode
                ).to(utilities.COMPUTE_DEVICE)

                # applying initialization method (Xavier for tanh, He for ReLU)
                nn_model.apply(nn_model.trainingMethod)

                # defining loss function and optimizer
                loss_function = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)

                # tracking training loss for each epoch
                training_loss_per_epoch = []

                # training the model over defined epochs
                for epoch in range(utilities.T):
                    epoch_loss = train_neural_network(
                        train_data_loader, nn_model, loss_function, optimizer
                    )
                    training_loss_per_epoch.append(epoch_loss)

                # evaluating the model on the test set
                test_loss = evaluate_neural_network(test_data_loader, nn_model, loss_function)

                # storing results 
                experiment_results[configuration_key] = {
                    "test_loss": test_loss,
                    "training_loss": training_loss_per_epoch,
                }

    print(f"Completed training for {total_combinations} different configurations.")

    # displaying the results
    print("\nFinal Results for Each Configuration:\n")
    for config, result in experiment_results.items():
        layer_width, network_depth, activation_mode = config
        activation_function = "ReLU" if activation_mode == 1 else "Tanh"

        average_training_loss = np.mean(result["training_loss"][-1])
        test_error = result["test_loss"]

        print(
            f"Configuration: Width={layer_width}, Depth={network_depth}, Activation={activation_function}"
        )
        print(f"\tFinal Training Error: {average_training_loss:.6f}")
        print(f"\tTest Error: {test_error:.6f}\n")

    print("Training and evaluation completed.")

if __name__ == "__main__":
    main()