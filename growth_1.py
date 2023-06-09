import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
alpha = 0.4
beta = 0.96
tol = 1e-6

# Discretized state space
k_min = 0.01
k_max = 2
n_k = 100
k_grid = np.linspace(k_min, k_max, n_k)

# Initial guess for value function
V_old = np.zeros(n_k)

while True:
    # Allocate memory to store new value function
    V_new = np.zeros(n_k)

    # Policy function
    policy = np.zeros(n_k)

    # Loop over all states
    for i in range(n_k):
        k = k_grid[i]

        # Objective function for minimization
        obj = lambda k_plus: -(np.log(k ** alpha - k_plus) + beta * np.interp(k_plus, k_grid, V_old))

        # Find the optimal k_plus given current k
        result = minimize(obj, x0=0.1, bounds=[(0, k ** alpha)])

        # Update maximum value and policy function
        V_new[i] = -result.fun
        policy[i] = result.x

    # Check the convergence criteria
    if np.max(np.abs(V_old - V_new)) < tol:
        break

    # Update value function
    V_old = V_new.copy()

# Plot value function
plt.figure()
plt.plot(k_grid, V_new)
plt.title('Value Function')
plt.show()

# Plot policy function
plt.figure()
plt.plot(k_grid, policy)
plt.title('Policy Function')
plt.show()
