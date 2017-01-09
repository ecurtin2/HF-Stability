import numpy as np
import tempfile
import sys
import os
from mpi4py import MPI


comm = MPI.COMM_WORLD
nprocs = comm.size
rank = comm.Get_rank()

def get_fname(rs, Nk, ndim):
    pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
    ext = '.log'
    outdir = 'log'
    fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)
    return fname

Nk = 20
#rsrange = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5]
rsrange = [1.2, 1.3, 1.4, 1.5]
ndim = 2
nguess = 1
mycase = "cRHF2cUHF"

paramlist = [[rs, Nk, mycase] for rs in rsrange]
fnames = [get_fname(paramlist[i][0], paramlist[i][1], ndim) for i in range(len(paramlist))]

for i in range(rank, len(paramlist), nprocs):
     cmd = './HFS ' + ' '.join([str(j) for j in paramlist[i]])
     print 'Starting Job: ', cmd
     os.system(cmd + ' > ' + fnames[i] + ' &')
