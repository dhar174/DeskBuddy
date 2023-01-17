import transformers
import json
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn


class MemoryNetwork(nn.Module):
    def __init__(self, input_size, output_size, memory_size, memory_vector_dim):
        super(MemoryNetwork, self).__init__()

        # Define the input and output layers
        self.input_layer = nn.Linear(input_size, memory_vector_dim)
        self.output_layer = nn.Linear(memory_vector_dim, output_size)

        # Define the memory component
        self.memory = nn.Linear(memory_vector_dim, memory_size)

        # Define the "controller" neural network
        self.controller = nn.LSTM(
            input_size + memory_vector_dim, memory_vector_dim)

    def forward(self, inputs, memory):
        # Transform the inputs using the input layer
        input_transformed = self.input_layer(inputs)

        # Use the controller to compute the next memory state
        output, (h_n, c_n) = self.controller(
            torch.cat([input_transformed, memory], dim=1))

        # Use the memory component to compute the memory vector
        memory_vector = self.memory(output)

        # Transform the memory vector using the output layer to generate the output
        output = self.output_layer(memory_vector)

        return output, memory_vector


class HierarchicalEncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(HierarchicalEncoderDecoder, self).__init__()

        # Define the lower-level encoder
        self.lower_encoder = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=True)

        # Define the higher-level encoder
        self.higher_encoder = nn.LSTM(
            2 * hidden_size, hidden_size, num_layers, bidirectional=True)

        # Define the decoder
        self.decoder = nn.LSTM(
            input_size + 2 * hidden_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # Encode the lower-level utterances using the lower-level encoder
        lower_encoded, _ = self.lower_encoder(inputs)

        # Encode the higher-level dialogue structure using the higher-level encoder
        higher_encoded, _ = self.higher_encoder(lower_encoded)

        # Decode the encoded inputs and dialogue structure using the decoder
        decoded, _ = self.decoder(torch.cat([inputs, higher_encoded], dim=2))

        # Transform the decoded outputs using the output layer
        output = self.output_layer(decoded)

        return output


class GraphNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_relations):
        super(GraphNetwork, self).__init__()

        # Define the input and output layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Define the "message-passing" layers
        self.edge_networks = nn.ModuleList(
            [nn.Linear(2 * hidden_size, hidden_size) for _ in range(num_relations)])
        self.node_networks = nn.ModuleList([nn.LSTM(
            input_size + hidden_size, hidden_size, num_layers) for _ in range(num_relations)])

        def forward(self, inputs, edges):
            # Transform the input nodes using the input layer
            nodes = self.input_layer(inputs)

            # Loop over the different relations in the graph
            for i, (edge_network, node_network) in enumerate(zip(self.edge_networks, self.node_networks)):
                # Compute the hidden states for the edges in this relation
                edge_hiddens = edge_network(
                    torch.cat([nodes[edges[:, 0]], nodes[edges[:, 1]]], dim=1))

                # Use the node network to propagate information between the nodes in this relation
                node_hiddens, _ = node_network(
                    torch.cat([inputs, edge_hiddens], dim=1))

                # Update the hidden states of the nodes
                nodes = node_hiddens

            # Transform the final node hidden states using the output layer
            output = self.output_layer(nodes)

            return output


class HierarchicalMemoryNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, memory_size, memory_vector_dim, device):
        super(HierarchicalMemoryNetwork, self).__init__()
        self.device = device
        self.memory_vector_dim = memory_vector_dim
        self.memory_size = memory_size
        # Define the lower-level encoder
        self.lower_encoder = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=True).to(device)

        # Define the higher-level encoder
        self.higher_encoder = nn.LSTM(
            2 * hidden_size, hidden_size, num_layers, bidirectional=True).to(device)

        # Define the memory component
        self.memory = nn.Linear(2 * hidden_size, memory_size).to(device)

        # Define the "controller" neural network
        self.controller = nn.LSTM(
            input_size + 2 * hidden_size + memory_vector_dim, hidden_size, num_layers).to(device)

        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, inputs, memory):
        # Encode the lower-level utterances using the lower-level encoder
        print("inputs", inputs.shape)
        print("actual inputs", inputs.dtype)
        print("type of original inputs",  type(inputs))  # torch.Tensor
        inputs.to(self.device)
        lower_encoded, _ = self.lower_encoder(inputs)

        # Encode the higher-level dialogue structure using the higher-level encoder
        higher_encoded, _ = self.higher_encoder(
            lower_encoded)

        # Use the memory component to compute the memory vector
        memory_vector = self.memory(
            higher_encoded.to(self.device)).to(self.device)
        print("memory_vector", memory_vector.shape)
        #memory_vector = memory_vector.unsqueeze(1).to(self.device)
        memory_vector = memory_vector.view(55, 1, self.memory_vector_dim)
        memory_vector.to(self.device)
        print("lower", lower_encoded.shape)
        print("memory_vector2", memory_vector.shape)
        print("catted",  torch.cat([inputs, higher_encoded.to(
            self.device), memory_vector], dim=2).to(self.device).shape)
        # Use the controller to compute the next memory state
        output, (h_n, c_n) = self.controller(
            torch.cat([inputs, higher_encoded.to(
                self.device), memory_vector], dim=2).to(self.device))

        # Transform the decoded outputs using the output layer
        output = self.output_layer(output)
        output.to(self.device)
        return output, memory_vector


class GraphMemoryNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_relations, memory_size, memory_vector_dim):
        super(GraphMemoryNetwork, self).__init__()

        # Define the input and output layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Define the "message-passing" layers
        self.edge_networks = nn.ModuleList(
            [nn.Linear(2 * hidden_size, hidden_size) for _ in range(num_relations)])
        self.node_networks = nn.ModuleList([nn.LSTM(
            input_size + hidden_size, hidden_size, num_layers) for _ in range(num_relations)])

        # Define the memory component
        self.memory = nn.Linear(2 * hidden_size, memory_size)

        # Define the "controller" neural network
        self.controller = nn.LSTM(
            input_size + hidden_size + memory_vector_dim, hidden_size, num_layers)

    def forward(self, inputs, edges, memory):
        # Transform the input nodes using the input layer
        nodes = self.input_layer(inputs)

        # Loop over the different relations in the graph
        for i, (edge_network, node_network) in enumerate(zip(self.edge_networks, self.node_networks)):
            # Compute the hidden states for the edges in this relation
            edge_hiddens = edge_network(
                torch.cat([nodes[edges[:, 0]], nodes[edges[:, 1]]], dim=1))

            # Use the node network to propagate information between the nodes in this relation
            node_hiddens, _ = node_network(
                torch.cat([inputs, edge_hiddens], dim=1))

            # Update the hidden states of the nodes
            nodes = node_hiddens

        # Use the memory component to compute the memory vector
        memory_vector = self.memory(nodes)

        # Use the controller to compute the next memory state
        output, (h_n, c_n) = self.controller(
            torch.cat([inputs, nodes, memory], dim=2))

        # Transform the decoded outputs using the output layer
        output = self.output_layer(output)

        return output, memory_vector


class HierarchicalGraphMemoryNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_relations, memory_size, memory_vector_dim):
        super(HierarchicalGraphMemoryNetwork, self).__init__()

        # Define the lower-level encoder
        self.lower_encoder = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=True)

        # Define the higher-level encoder
        self.higher_encoder = nn.LSTM(
            2 * hidden_size, hidden_size, num_layers, bidirectional=True)

        # Define the input and output layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Define the "message-passing" layers
        self.edge_networks = nn.ModuleList(
            [nn.Linear(2 * hidden_size, hidden_size) for _ in range(num_relations)])
        self.node_networks = nn.ModuleList([nn.LSTM(
            input_size + hidden_size, hidden_size, num_layers) for _ in range(num_relations)])

        # Define the memory component
        self.memory = nn.Linear(2 * hidden_size, memory_size)

        # Define the "controller" neural network
        self.controller = nn.LSTM(
            input_size + 2 * hidden_size + memory_vector_dim, hidden_size, num_layers)

    def forward(self, inputs, edges, memory):
        # Encode the lower-level utterances using the lower-level encoder
        lower_encoded, _ = self.lower_encoder(inputs)

        # Encode the higher-level dialogue structure using the higher-level encoder
        higher_encoded, _ = self.higher_encoder(lower_encoded)

        # Transform the input nodes using the input layer
        nodes = self.input_layer(inputs)

        # Loop over the different relations in the graph
        for i, (edge_network, node_network) in enumerate(zip(self.edge_networks, self.node_networks)):
            # Compute the hidden states for the edges in this relation
            edge_hiddens = edge_network(
                torch.cat([nodes[edges[:, 0]], nodes[edges[:, 1]]], dim=1))

            # Use the node network to propagate information between the nodes in this relation
            node_hiddens, _ = node_network(
                torch.cat([inputs, edge_hiddens], dim=1))

            # Update the hidden states of the nodes
            nodes = node_hiddens

        # Use the memory component to compute the memory vector
        memory_vector = self.memory(nodes)

        # Use the controller to compute the next memory state
        output, (h_n, c_n) = self.controller(
            torch.cat([inputs, higher_encoded, memory], dim=2))

        # Transform the decoded outputs using the output layer
        output = self.output_layer(output)

        return output, memory_vector


json_file = os.getcwd() + '/training_data.json'


class MyDataset(Dataset):

    def __init__(self, json_file):
        fart = 0

        # Open the JSON file
        # Load the tokenizer from the model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            'facebook/blenderbot-400M-distill')
        self.padding_value = torch.tensor(
            self.tokenizer.pad_token_id, dtype=torch.float32)

        # Define the input and target tokenizers

        self.data = []
        with open(json_file, 'r') as f:
            data_dict = json.load(f)
            self.max_length = 128

            for i in range(len(data_dict)):
                if data_dict[i] != " " or data_dict[i] != "":
                    # print(data_dict[i])
                    text = data_dict[i]
                    if text == " " or text == "" or data_dict[i] == " " or data_dict[i] == "":
                        continue
                    if i % 2 == 0:
                        input_str = data_dict[i]
                        # remove "Input: " prefix and quotation marks

                    else:
                        target_str = data_dict[i]
                        # remove "Target: " prefix and quotation marks
                        self.data.append((input_str, target_str))
                        # print("Input and target:", input_str, target_str)

    def __getitem__(self, index):
        data_dict = self.data[index]
        i = 0
        input_tokens = []
        target_tokens = []
        print("index", index)
        print("len", len(self.data))
        print("data_dict", data_dict)
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range")
        else:
            print("i", i)
            input_str = data_dict[0]
            if input_str.startswith('Input (from Emma)'):
                input_tokens.append(self.tokenizer.encode(
                    input_str, add_special_tokens=True, max_length=self.max_length, truncation=True))
            else:
                input_tokens.append(self.tokenizer.encode(
                    input_str, add_special_tokens=True, max_length=self.max_length, truncation=True))
                print("Check input_str", input_str)
            target_str = data_dict[1]
            if target_str.startswith('Target (from Picoh)'):
                target_tokens.append(self.tokenizer.encode(
                    target_str, add_special_tokens=True, max_length=self.max_length, truncation=True))
            else:
                target_tokens.append(self.tokenizer.encode(
                    target_str, add_special_tokens=True, max_length=self.max_length, truncation=True))
                print("Check target_str", target_str)

            i += 1

            if input_tokens and target_tokens:
                # Convert input and target tokens to tensors and pad them
                padded_inputs = self.pad_sequences(
                    input_tokens,  max_len=self.max_length)
                padded_targets = self.pad_sequences(
                    target_tokens, max_len=self.max_length)
                # Convert padded input and target tensors back to lists of integers
                # print("input_tokens", len(padded_inputs[0]))
                # print("target tokens",  len(padded_targets[0]))
                inputs = padded_inputs
                targets = padded_targets
                # Convert the lists of integers to tensors
                inputs = torch.tensor(inputs, dtype=torch.float32)
                print("inputs", inputs.dtype)
                targets = torch.tensor(targets, dtype=torch.float32)
                print("targets", targets.dtype)

        return inputs, targets

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def pad_sequences(self, sequences, max_len):

        padded_sequences = []
        for sequence in sequences:
            if len(sequence) < max_len:
                padded_sequence = sequence + \
                    [self.padding_value] * (max_len - len(sequence))
            else:
                padded_sequence = sequence[:max_len]
            padded_sequences.append(padded_sequence)
        return padded_sequences


# Load the training dataset
train_dataset = MyDataset(json_file)
# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
device = 'cpu'

# Initialize the memory to None
memory_network = HierarchicalMemoryNetwork(128, 128, 128, 2, 8, 128, device)
memory_network.memory = nn.Linear(2 * 128, 8).to(device)

memory_network = memory_network.to(device)
# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(memory_network.parameters())
# # Iterate over the training dataset
# for inputs, targets in dataset:
#     # Forward pass
#     output, memory = memory_network(inputs, memory)
#     # Compute the loss
#     loss = loss_fn(output, targets)
#     # Backward pass
#     loss.backward()
#     # Update the weights
#     optimizer.step()


class TrainingLoop:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device=device, patience=5):
        self.patience = patience

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.memory = None

    def train(self, epochs):
        best_val_loss = float('inf')
        wait = 0
        for epoch in range(epochs):
            self.model.to(self.device)
            self.model.train()

            batch_idx = 0
            for inputs, targets in train_loader:
                inputs.to(self.device)
                targets.to(self.device)
                self.model.memory.to(self.device)
                # Forward pass
                output, memory = memory_network(inputs, self.model.memory)
                self.model.memory = memory
                output.to(self.device)
                # Compute the loss
                loss = loss_fn(output, targets)
                # Backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                batch_idx += 1
                # Print the progress
                if (batch_idx % 100 == 0):
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx *
                        len(inputs), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))
            with torch.no_grad():
                val_loss = 0
                for inputs, targets in self.val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    output, _ = self.model(inputs, memory)
                    val_loss += self.loss_fn(output, targets).item()
                val_loss /= len(self.val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait > self.patience:
                        return
            self.model.eval()
            torch.save(self.model.state_dict(), 'trained_model.pth')
