import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the Branin function
def branin(X):
    x1, x2 = X
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

# Generate grid data for Branin function
x = np.linspace(-5, 10, 400)
y = np.linspace(0, 15, 400)
X, Y = np.meshgrid(x, y)
Z_branin = branin([X, Y])

# Define the three minima of the Branin function
branin_minima = [(9.42478, 2.475), (-np.pi, 12.275), (np.pi, 2.275)]

# Load the CSV file for the heatmap
file_path = 'mean_iterations_minus_five.csv'  # Adjust path as needed
data = pd.read_csv(file_path)

# Pivot the data to create a matrix with 'batch_size' as rows and 'initial_samples' as columns
heatmap_data = data.pivot(index='batch_size', columns='initial_samples', values='mean_iterations')

# Sort the rows (batch_size) and columns (initial_samples) in increasing order
heatmap_data = heatmap_data.sort_index(ascending=True).sort_index(axis=1, ascending=True)

# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Plot A: Branin function contour plot with minima
contour_branin = ax[0].contour(X, Y, Z_branin, 15, cmap='viridis')
ax[0].set_title('A: Branin Function Contour Plot', fontsize=16, loc='left')
ax[0].set_xlabel('x1', fontsize=14)
ax[0].set_ylabel('x2', fontsize=14)
fig.colorbar(contour_branin, ax=ax[0])
for minima in branin_minima:
    ax[0].plot(minima[0], minima[1], 'ro', markersize=8)

# Plot B: Heatmap of mean iterations
sns.heatmap(heatmap_data, annot=False, cmap='magma', cbar=True, fmt='.1f', linewidths=0.2, linecolor='black', ax=ax[1])
ax[1].invert_yaxis()
ax[1].set_title('B: Heatmap of Mean Iterations to Reach Maximum', fontsize=16, loc='left')
ax[1].set_xlabel('Initial Samples', fontsize=14)
ax[1].set_ylabel('Batch Size', fontsize=14)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig("combined_branin_heatmap.png", dpi=600, transparent=True, bbox_inches="tight")
plt.show()
