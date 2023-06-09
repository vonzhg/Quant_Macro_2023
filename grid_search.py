import torch


def solve_z(x, y, b, z_min, z_max, n_z):
    # Create a grid for z values
    z_grid = torch.linspace(z_min, z_max, n_z)

    # Expand dimensions for broadcasting
    z_grid = z_grid.unsqueeze(0).expand(x.size(0), -1)  # Shape: [n, n_z]
    x = x.unsqueeze(1).expand(-1, n_z)  # Shape: [n, n_z]
    y = y.unsqueeze(1).expand(-1, n_z)  # Shape: [n, n_z]

    # Compute the absolute value of the function for each x, y, z
    abs_val = torch.abs(x * z_grid ** 5 + y * z_grid ** 3 + b)

    # Find the index of z in the grid that minimizes the value for each x, y
    min_val, min_indices = torch.min(abs_val, dim=1)

    # Get the corresponding z values
    z_sol = z_grid[torch.arange(x.size(0)), min_indices]

    return z_sol, min_val


# Usage:
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
b = 10
z_min = -10
z_max = 10
n_z = 1000

z_sol, min_val = solve_z(x, y, b, z_min, z_max, n_z)
print("Solved z values:", z_sol)
print("Minimum absolute values:", min_val)
