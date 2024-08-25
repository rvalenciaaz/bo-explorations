import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the kernel: a product of a constant kernel and an RBF kernel
kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))

# Create GaussianProcessRegressor object
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Generate a different set of 4 random training data points
np.random.seed(24)
X_train = np.random.uniform(-3., 3., 4)[:, np.newaxis]
y_train = np.sin(X_train).ravel()

# Fit to the data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)

# Make predictions and also draw samples from the GP
X = np.linspace(-5, 5, 1000)[:, np.newaxis]
y_pred, sigma = gp.predict(X, return_std=True)
samples = gp.sample_y(X, 3)  # draw 3 samples from the Gaussian Process

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'r.', markersize=10) 
#plt.errorbar(X_train, y_train, yerr=0.1, fmt='r.', markersize=10)
plt.plot(X, y_pred, 'b-')
plt.fill_between(X.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k')
plt.plot(X, samples, 'k--', lw=1)
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Gaussian Process Regression')
plt.savefig('gaussian_process_regression_plot_no_legend.png')
plt.close()
