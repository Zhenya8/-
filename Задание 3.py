from mpi4py import MPI
import numpy as np
import time


def f(x, y):
    return np.sin(x) * np.cos(y)


def df_dx(grid, h):
    nx, ny = grid.shape
    derivative_grid = np.zeros_like(grid)

    for i in range(1, nx - 1):
        for j in range(ny):
            derivative_grid[i][j] = (grid[i + 1][j] - grid[i - 1][j]) / (2 * h)

    return derivative_grid


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

grid_sizes = [(10, 10), (100, 100), (1000, 1000)]

for grid_size in grid_sizes:
    nx, ny = grid_size
    h = 0.1

    x_values = np.linspace(0, 2 * np.pi, nx)
    y_values = np.linspace(0, 2 * np.pi, ny)
    grid = np.zeros((nx, ny))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            grid[i][j] = f(x, y)

    chunk_size = nx // size
    local_grid = np.zeros((chunk_size, ny))
    comm.Scatter(grid, local_grid, root=0)

    start_time = time.time()
    local_derivative = df_dx(local_grid, h)
    global_derivative = np.zeros_like(grid)
    comm.Gather(local_derivative, global_derivative, root=0)
    end_time = time.time()

    if rank == 0:
        print(f"Grid size: {nx}x{ny}, Time taken: {
            end_time - start_time} seconds")
