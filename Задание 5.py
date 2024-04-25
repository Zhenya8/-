from mpi4py import MPI
import numpy as np
import time


def generate_random_system(size):
    A = np.random.rand(size, size)
    b = np.random.rand(size)
    return A, b


def gaussian_elimination(A, b):
    n = len(b)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

system_sizes = [10, 100, 1000]

for system_size in system_sizes:
    A, b = generate_random_system(system_size)

    chunk_size = system_size // size
    local_A = np.zeros((chunk_size, system_size))
    comm.Scatter(A, local_A, root=0)
    local_b = np.zeros(chunk_size)
    comm.Scatter(b, local_b, root=0)

    start_time = time.time()
    solution = gaussian_elimination(local_A, local_b)
    global_solution = np.zeros(system_size)
    comm.Gather(solution, global_solution, root=0)
    end_time = time.time()

    if rank == 0:
        print(f"System size: {system_size}, Time taken: {
            end_time - start_time} seconds")
