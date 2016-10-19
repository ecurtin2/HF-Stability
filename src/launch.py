import _HFS as HFS
import numpy as np
import tempfile


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


Nkrange = range(20,1000)
rs = 1.2
ndim = 2

paramlist = [[rs, i, ndim, get_fname(rs, i, ndim)] for i in Nkrange]
for i in range(my_rank, len(paramlist), nprocs):
    #print "rank", my_rank, "rs=%f Nk=%d ndim=%d fname=%s" % tuple(paramlist[i])
    HFS.main_(*paramlist[i])
