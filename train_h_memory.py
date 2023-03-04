import os
import dialogue_management as dm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
json_file = os.getcwd() + '/training_data.json'
device = dm.device

model = dm.HierarchicalMemoryNetwork(128, 128, 128, 2, 8, 128, device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
train_loader = DataLoader(dm.MyDataset(json_file), batch_size=32, shuffle=True)
val_loader = DataLoader(dm.MyDataset(json_file), batch_size=32, shuffle=False)

# Initialize the TrainingLoop
training_loop = dm.TrainingLoop(
    model, optimizer, loss_fn, train_loader, val_loader, device)

# Train the model
training_loop.train(10)

print("Training complete. Saving model...")
