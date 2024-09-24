import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

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

# Define the Ackley function
def ackley(X):
    x1, x2 = X
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

# Generate grid data for Branin function
x = np.linspace(-5, 10, 400)
y = np.linspace(0, 15, 400)
X, Y = np.meshgrid(x, y)
Z_branin = branin([X, Y])

# Generate grid data for Ackley function
x_ack = np.linspace(-5, 5, 400)
y_ack = np.linspace(-5, 5, 400)
X_ack, Y_ack = np.meshgrid(x_ack, y_ack)
Z_ackley = ackley([X_ack, Y_ack])

# Define the three minima of the Branin function
branin_minima = [(9.42478, 2.475), (-pi, 12.275), (pi, 2.275)]

# Plotting with colored level lines, indicating the minima, and increasing point size
plt.figure(figsize=(14, 6))

# Branin function contour plot with colored level lines and minima
plt.subplot(1, 2, 1)
contour_branin = plt.contour(X, Y, Z_branin, 15, cmap='viridis')
plt.colorbar(contour_branin)
for minima in branin_minima:
    plt.plot(minima[0], minima[1], 'ro', markersize=8)  # Increased size of the minima points
plt.title('Branin Function Contour Plot')
plt.xlabel('x1')
plt.ylabel('x2')

# Ackley function contour plot with colored level lines and minimum
plt.subplot(1, 2, 2)
contour_ackley = plt.contour(X_ack, Y_ack, Z_ackley, 15, cmap='viridis')
plt.colorbar(contour_ackley)
plt.plot(0, 0, 'ro', markersize=8)  # Increased size of the minima point
plt.title('Ackley Function Contour Plot')
plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.savefig('branin_ackley_colored_level_lines_minima.png',dpi=600, transparent=True)
plt.show()
