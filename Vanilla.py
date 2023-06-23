import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load and preprocess the training data
# Assuming you have training data X_train

# Normalize the input data between 0 and 1
X_train = X_train.astype('float32') / 255.

# Flatten the input data if needed
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))

# Convert the data to torch tensors
X_train = torch.from_numpy(X_train)

# Create an instance of the autoencoder
autoencoder = Autoencoder()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(autoencoder.parameters())

# Training loop
num_epochs = 10
batch_size = 256

for epoch in range(num_epochs):
    # Mini-batch training
    for i in range(0, X_train.size(0), batch_size):
        inputs = X_train[i:i+batch_size]
        
        # Forward pass
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print the loss after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Obtain the encoded representation of the input data
encoded_imgs = autoencoder.encoder(X_train).detach().numpy()

# Obtain the decoded representation of the encoded data
decoded_imgs = autoencoder.decoder(torch.from_numpy(encoded_imgs)).detach().numpy()

# Use the encoded or decoded representation for further tasks if needed
