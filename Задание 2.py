import time

import numpy as np
from mpi4py import MPI


def sum_array(numbers):
    return np.sum(numbers)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

array_sizes = [10, 1000, 10000000]

for array_size in array_sizes:
    numbers = np.arange(array_size, dtype=np.float64)

    chunk_size = len(numbers) // size
    local_numbers = np.zeros(chunk_size, dtype=np.float64)
    comm.Scatter(numbers, local_numbers, root=0)

    start_time = time.time()
    local_sum = sum_array(local_numbers)
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    end_time = time.time()

    if rank == 0:
        print(f"Array size: {array_size}, Time taken: {
              end_time - start_time} seconds, Global sum: {global_sum}")
