import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool

# Parameters
alpha = 0.4
beta = 0.96
tol = 1e-2

# Discretized state space
k_min = 0.01
k_max = 2
n_k = 100
k_grid = np.linspace(k_min, k_max, n_k)

# Initial guess for value function
V_old = np.zeros(n_k)

def max_value_and_policy(i):
    k = k_grid[i]
    max_value = -1e10
    policy_k_plus = 0.0

    # Loop over all possible k'
    for j in range(n_k):
        k_plus = k_grid[j]

        # Check the budget constraint
        if k ** alpha - k_plus >= 0:
            # Consumption
            c = k ** alpha - k_plus

            # Value of this choice
            value = np.log(c) + beta * V_old[j]

            # Update maximum value and policy function
            if value > max_value:
                max_value = value
                policy_k_plus = k_plus

    return max_value, policy_k_plus

if __name__ == '__main__':
    p = Pool()
    print(f'Using {p.ncpus} processes')

    max_iters = 1000  # set a maximum number of iterations
    iters = 0

    while True:
        # Allocate memory to store new value function
        V_new = np.zeros(n_k)

        # Policy function
        policy = np.zeros(n_k)

        # Use a pool of workers
        results = p.map(max_value_and_policy, range(n_k))

        # Unpack results
        for i, result in enumerate(results):
            V_new[i], policy[i] = result

        # Check the convergence criteria
        max_diff = np.max(np.abs(V_old - V_new))
        print(f"Iteration {iters}, maximum difference: {max_diff}")
        if max_diff < tol:
            break

        # Update value function
        V_old = V_new.copy()

        iters += 1
        if iters > max_iters:
            print("Reached maximum number of iterations without convergence.")
            break

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
