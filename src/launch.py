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

Nkrange = range(10, 20)
rs = 1.2
ndim = 2
nguess = 50
blocksize = 20
num_evals = 8
maxits = 50
maxsubsize = 1000
tolerance = 1e-8


paramlist = [['./HFS', rs, Nk, ndim, get_fname(rs, Nk, ndim), tolerance, maxits,
              maxsubsize, nguess, blocksize, num_evals] for Nk in Nkrange]

for i in range(my_rank, len(paramlist), nprocs):
     print 'Starting Job: ',  ' '.join([str(j) for j in paramlist[i]])
     os.system(' '.join([str(j) for j in paramlist[i]]))
