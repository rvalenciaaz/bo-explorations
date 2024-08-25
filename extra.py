import torch
import torch.nn as nn
import torch.distributions as dist


# Check for CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define an MLP architecture
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
# Define a function for training an MLP
def train_mlp(model, criterion, optimizer, train_X, train_Y, epochs=500):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_Y)
        loss.backward()
        optimizer.step()


class BayesianModelAveragingMLP(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        # Initialize weights uniformly
        self.weights = torch.nn.Parameter(torch.ones(len(models)) / len(models), requires_grad=True)
        # Learnable parameter for additional uncertainty (aleatoric)
        self.log_noise = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, X):
        with torch.no_grad():
            # Predictions from each model
            preds = torch.stack([model(X) for model in self.models], dim=0)
            # Weighted mean across the models
            mean = torch.einsum('i,ijk->jk', torch.softmax(self.weights, dim=0), preds)
            # Variance as a weighted combination of model variance and additional noise
            variance = preds.var(dim=0) + torch.exp(self.log_noise)
        # Returning a Gaussian distribution with the computed mean and variance
        return dist.Normal(mean, variance.sqrt())

# Generating synthetic training data
train_X = torch.linspace(0, 1, 100).unsqueeze(-1).to(device)
train_Y = torch.sin(train_X * (2 * torch.pi)) + torch.randn_like(train_X) * 0.2

# Initialize and train MLP models
input_size = train_X.shape[1]
hidden_size = 10  # Example hidden layer size
output_size = 1
model1 = SimpleMLP(input_size, hidden_size, output_size).to(device)
model2 = SimpleMLP(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

train_mlp(model1, criterion, optimizer1, train_X, train_Y, epochs=1000)
train_mlp(model2, criterion, optimizer2, train_X, train_Y, epochs=1000)
# Example usage with BayesianModelAveragingMLP
# Assuming models are defined and trained, and test_X is prepared

bma_model = BayesianModelAveragingMLP(models=[model1, model2]).to(device)

# Since we're working with a distribution, let's see how to use it
test_X = torch.linspace(0, 1, 50).unsqueeze(-1).to(device)
predictive_distribution = bma_model(test_X)

# Sample from the predictive distribution or calculate confidence intervals, etc.
# Example: Extract mean and standard deviation for each point in test_X
means = predictive_distribution.mean
stddevs = predictive_distribution.stddev
