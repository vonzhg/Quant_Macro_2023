import numpy as np
import matplotlib.pyplot as plt

def solve_quadratic(a, b):
    discriminant = b**2 - 4 * a * 10
    if discriminant >= 0:
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return root1, root2
    else:
        return None

# Define the range of values for a and b
a_values = np.linspace(2, 3, 100)
b_values = np.linspace(9, 10, 100)

roots = np.zeros((len(a_values), len(b_values), 2))
analytical_roots = []

# Solve the quadratic equation for different values of a and b
for i, a in enumerate(a_values):
    for j, b in enumerate(b_values):
        root = solve_quadratic(a, b)
        if root is not None:
            roots[i, j] = root
            analytical_root = (-b + np.sqrt(b**2 - 4 * a * 10)) / (2 * a)
            analytical_roots.append(analytical_root)

roots = roots.reshape(-1, 2)
analytical_roots = np.array(analytical_roots)

# Plot the solutions against the values of a and b
plt.scatter(a_values.repeat(len(b_values)), np.tile(b_values, len(a_values)), c=roots[:, 0], cmap='coolwarm', label='Root 1')
plt.scatter(a_values.repeat(len(b_values)), np.tile(b_values, len(a_values)), c=roots[:, 1], cmap='coolwarm', label='Root 2')
plt.colorbar(label='Root Value')
plt.xlabel('a')
plt.ylabel('b')
plt.title('Solutions of Quadratic Equation')
plt.legend()
plt.show()
