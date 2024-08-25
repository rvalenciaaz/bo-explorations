import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Define the kernel and the Gaussian Process model
kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel)

# Create the test points where we want to evaluate the functions
X = np.linspace(-5, 5, 100).reshape(-1, 1)

# Generate the mean and covariance of the GP at the test points
mean, std = gp.predict(X, return_std=True)

# Draw multiple samples from the Gaussian Process
samples = gp.sample_y(X, n_samples=5, random_state=42)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X, samples, lw=2)  # Plot the samples
plt.fill_between(X.ravel(), mean - 1.96*std, mean + 1.96*std, alpha=0.2, color='gray')  # Plot the confidence interval
plt.plot(X, mean, 'k--', lw=2, label='Mean')  # Plot the mean function
plt.title("Gaussian Process example")
plt.xlabel("X")
plt.ylabel("f(X)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
# Save the plot as an image file
plt.savefig('gaussian_process_plot.png',dpi=600,bbox_inches="tight")
plt.close()
