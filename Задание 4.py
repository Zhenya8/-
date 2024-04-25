from mpi4py import MPI
import numpy as np
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
matrix_sizes = [(10, 10), (100, 100), (1000, 1000)]

for matrix_size in matrix_sizes:
    nrows, ncols = matrix_size
    A = np.random.rand(nrows, ncols)
    B = np.random.rand(ncols, nrows)
    chunk_size = nrows // size
    local_A = np.zeros((chunk_size, ncols))
    comm.Scatter(A, local_A, root=0)
    local_B = np.zeros((ncols, nrows))
    comm.Bcast(B, root=0)
    start_time = time.time()
    local_C = matrix_multiply(local_A, B)
    global_C = np.zeros((nrows, nrows))
    comm.Gather(local_C, global_C, root=0)
    end_time = time.time()
    if rank == 0:
        print(f"Matrix size: {nrows}x{ncols}, Time taken: {
            end_time - start_time} seconds")
