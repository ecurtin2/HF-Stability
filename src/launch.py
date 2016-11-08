import numpy as np
import tempfile
import sys
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.size
my_rank = comm.Get_rank()


def get_fname(rs, Nk, ndim):
    pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
    ext = '.log'
    outdir = 'log'
    fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)
    return fname

Nk = 57
rsrange = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5]
ndim = 2
nguess = 1
blocksize = 1
num_evals = 1
maxits = 50
maxsubsize = 1000
tolerance = 1e-6


paramlist = [['./HFS', rs, Nk, ndim, get_fname(rs, Nk, ndim), tolerance, maxits,
              maxsubsize, nguess, blocksize, num_evals] for rs in rsrange]

for i in range(my_rank, len(paramlist), nprocs):
     print 'Starting Job: ',  ' '.join([str(j) for j in paramlist[i]])
     os.system(' '.join([str(j) for j in paramlist[i]]))
