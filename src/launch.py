import _HFS as HFS
import numpy as np

"""
from mpi4py import MPI

comm = MPI.MPI_COMM_WORLD
nprocs = comm.Get_size()
my_rank = comm.Get_rank()

nk = np.uint64(11)

paramlist = [x, y, z, "out%d.dat" % my_rank ]
for i in range(my_rank, len(paramlist), nprocs):
    HFS.main_(*paramlist[i])

HFS.main_(1.2, 11, 2)
"""

for Nk in range(10, 20):
    HFS.main_(1.2, Nk, 2)
