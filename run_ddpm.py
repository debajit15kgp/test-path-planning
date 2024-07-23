import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
import numpy as np

# 1. Define the data
def generate_circle_data(num_points=8, radius=1, noise=0.05):
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = radius * np.cos(theta) + np.random.normal(0, noise, num_points)
    y = radius * np.sin(theta) + np.random.normal(0, noise, num_points)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

data = generate_circle_data()
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. Define the model
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t):
        # t is not used in this simple model, but would be in a full implementation
        return self.net(x)

model = SimpleUNet()

# 3. Define the diffusion process
num_train_steps = 1000
noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_steps)

# 4. Training loop
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10000

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Add noise to the data
        noise = torch.randn_like(batch[0])
        timesteps = torch.randint(0, num_train_steps, (batch[0].shape[0],))
        noisy_samples = noise_scheduler.add_noise(batch[0], noise, timesteps)

        # Predict the noise
        noise_pred = model(noisy_samples, timesteps)

        # Compute loss
        loss = nn.MSELoss()(noise_pred, noise)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 5. Sampling from the model
@torch.no_grad()
def sample(num_samples=1000):
    # Start from Gaussian noise
    x = torch.randn(num_samples, 2)

    for t in reversed(range(num_train_steps)):
        timesteps = torch.full((num_samples,), t, dtype=torch.long)
        noise_pred = model(x, timesteps)
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x

# Generate samples
samples = sample().numpy()

# 6. Visualize results
plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], c='r', label='Original data')
plt.scatter(samples[:, 0], samples[:, 1], c='b', alpha=0.1, label='Generated samples')
plt.legend()
plt.title("Original Data vs Generated Samples")
plt.axis('equal')
plt.show()