import torch
import torch.nn as nn
import torch.optim as optim

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

# Adapt BayesianModelAveraging class for MLPs
class BayesianModelAveragingMLP(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.weights = torch.ones(len(models), device=device) / len(models)
    
    def fit(self, likelihoods):
        self.weights = torch.softmax(torch.tensor(likelihoods, device=device), dim=0)
    
    def forward(self, X):
        preds = [model(X) for model in self.models]
        mean = torch.stack(preds, dim=0).mean(dim=0)
        variance = torch.stack(preds, dim=0).var(dim=0)
        return mean, variance

# Use BayesianModelAveragingMLP
bma_model = BayesianModelAveragingMLP(models=[model1, model2])
bma_model.to(device)
bma_model.fit([1.0, 1.5])  # Example likelihoods

# Make predictions
test_X = torch.linspace(0, 1, 50).unsqueeze(-1).to(device)
mean, variance = bma_model(test_X)

print(mean, variance)
# The mean and variance are the model's predictions and estimated uncertainties
