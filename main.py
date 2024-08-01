import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class VectorDataset(Dataset):
    def __init__(self, input_csv, output_csv):
        # Load data from CSV files
        self.input_data = np.transpose(pd.read_csv(input_csv).values)
        self.output_data = np.transpose(pd.read_csv(output_csv).values)

    def __len__(self):
        # Return the number of samples
        return len(self.input_data)

    def __getitem__(self, idx):
        # Retrieve the sample at the given index
        x = self.input_data[idx]
        y = self.output_data[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# A simple example
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(90, 120)  # First layer
        self.fc2 = nn.Linear(120, 60)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)
        return x


# We can also build a more complex network
class ComplexRegressionNet(nn.Module):
    def __init__(self):
        super(ComplexRegressionNet, self).__init__()
        # Define the architecture here
        self.fc1 = nn.Linear(90, 256)  # First hidden layer
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization for first hidden layer
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization for second hidden layer
        self.fc3 = nn.Linear(128, 128)  # Third hidden layer
        self.dp1 = nn.Dropout(0.5)  # Dropout layer
        self.fc4 = nn.Linear(128, 64)  # Fourth hidden layer
        self.fc5 = nn.Linear(64, 60)  # Output layer

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # Activation function is ReLU
        x = F.relu(self.bn2(self.fc2(x)))  # ReLU + BatchNorm on second layer
        x = F.relu(self.fc3(x))  # ReLU on third layer
        x = self.dp1(x)  # Apply dropout
        x = F.relu(self.fc4(x))  # ReLU on fourth layer
        x = self.fc5(x)  # Output layer, no activation
        return x


# Assuming your data is stored in 'input_data.csv' and 'output_data.csv'
input_csv = 'data/M1.csv'
output_csv = 'data/A1.csv'

# Create the dataset
dataset = VectorDataset(input_csv, output_csv)

# Calculate the size of the dataset
total_size = len(dataset)

# Assuming train_size is defined and you want the remainder for test_size
train_size = int(total_size*0.9)
test_size = total_size - train_size

# Create the training dataset and test dataset by slicing
train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the network
net = ComplexRegressionNet()

# Print the network architecture (optional)
print(net)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define the scheduler
# Assuming you want to decay the learning rate by a factor of 0.1 every 200 epochs
scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

# Number of epochs
epochs = 1000  # For example, if you want to train for a long time

# Let's assume you have two lists that contain the losses
train_losses = []
test_losses = []

for epoch in range(epochs):  # loop over the dataset multiple times
    # Training phase
    net.train()  # Set the model to training mode
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()

    # Record the average training loss for this epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Testing phase
    net.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Record the average testing loss for this epoch
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}, Testing Loss: {avg_test_loss}")

print('Finished Training')

# Save the model checkpoint
torch.save(net.state_dict(), 'model.pth')

# # After training, plot the training and testing losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.title('Training and Testing Losses Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the figure without displaying it
plt.savefig('loss_plot.png')

# Load the model checkpoint
net.load_state_dict(torch.load('model.pth'))
net.eval()  # Set the model to evaluation mode

# Plotting the testing results
# Select one example to plot
example_index = 0  # For instance, the first example in the test set
inputs, actual_output = next(iter(test_loader))  # Get the first batch
with torch.no_grad():  # Disable gradient calculation
    predicted_output = net(inputs)  # Get the model's predictions

# Convert the tensors to numpy arrays if they're not already
actual_output_np = actual_output.numpy()
predicted_output_np = predicted_output.numpy()

# Now plot the actual vs predicted output for the selected example
plt.figure(figsize=(10, 5))
plt.plot(actual_output_np[example_index], label='Actual Output')
plt.plot(predicted_output_np[example_index], label='Predicted Output')
plt.title('Comparison of Actual and Predicted Outputs of Index {}'.format(example_index))
plt.xlabel('Output Points')
plt.ylabel('Value')
plt.legend()
plt.show()

# Save the figure without displaying it
plt.savefig('Testing results of index {}.png'.format(example_index))