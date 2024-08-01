import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# I forget the input sizes and I assume it is 250 \times 150
# Please change it to match the real data size
# The Unet has downsampling so this dimension should be large enough to
# repeatedly be divided by 2.
Input_width=250
Input_Length=150
# I forget the input sizes and I assume it is 53
# Feel free to change it
Output_Length=60

# Custom Dataset class
class Matirx2VectorDataset(Dataset):
    def __init__(self, input_csv, output_csv):
        # Load data from CSV files
        self.input_data = pd.read_csv(input_csv).values.reshape(-1, 1, Input_width, Input_Length)  # Reshape for image input
        self.output_data = pd.read_csv(output_csv).values.reshape(-1, Output_Length)
        # Make sure they have a same dimension. Here has to be very careful.
        if len(self.input_data)<len(self.output_data):
            self.output_data=self.output_data[:len(self.input_data)]
        else:
            self.input_data=self.input_data[:len(self.output_data)]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.output_data[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# U-Net building block: Conv -> BatchNorm -> ReLU
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# Revised U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)

        # Decoder
        self.dec4 = UNetBlock(1024, 512)
        self.dec3 = UNetBlock(512, 256)
        self.dec2 = UNetBlock(256, 128)
        self.dec1 = UNetBlock(128, 64)

        # Final output layer for 1D vector
        # Final layers for 1D vector output (flattened)
        self.fc1 = None  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(1024, Output_Length)  # Output layer

    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))

        # Bottleneck
        x5 = self.bottleneck(F.max_pool2d(x4, 2))

        # Bottleneck
        x5 = self.bottleneck(F.max_pool2d(x4, 2))

        # Decoder path
        x4d = self.dec4(F.interpolate(x5, scale_factor=2))
        x3d = self.dec3(F.interpolate(x4d, scale_factor=2))
        x2d = self.dec2(F.interpolate(x3d, scale_factor=2))
        x1d = self.dec1(F.interpolate(x2d, scale_factor=2))

        # Flatten and dynamically define the fully connected layer if not already defined
        x1d_flat = x1d.view(x1d.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x1d_flat.size(1), 1024).to(x1d_flat.device)

        x_fc1 = F.relu(self.fc1(x1d_flat))
        output = self.fc2(x_fc1)

        return output

        return output

# Assuming your data is stored in 'input_data.csv' and 'output_data.csv'
input_csv = 'data/M1.csv'
output_csv = 'data/A1.csv'

# Create the dataset
dataset = Matirx2VectorDataset(input_csv, output_csv)

# Calculate the size of the dataset
total_size = len(dataset)

# Assuming train_size is defined and you want the remainder for test_size
train_size = int(total_size * 0.9)
test_size = total_size - train_size

# Create the training dataset and test dataset by slicing
train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the network
net = UNet(in_channels=1)

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
epochs = 1  # For example, if you want to train for a long time

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
