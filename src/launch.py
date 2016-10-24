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


Nkrange = range(10,250)
rs = 1.2
ndim = 2
Nk = 24
nguess = 80
blocksize = 20 # overwritten
num_evals = 5  # overwritten
minits = 5
maxits = 30
maxsubsize = 3000
tolerance = 1e-8


paramlist = [[rs, Nk, ndim, nguess, blocksize, num_evals, minits,  maxits, maxsubsize, tolerance, get_fname(rs, Nk, ndim)] for Nk in Nkrange]
# int main_(double rs, int Nk, int ndim, int num_guess_vecs, int dav_blocksize, int num_evals, int minits, int maxits, int maxsubsize, double tol, std::string outputfilename)


for i in range(my_rank, len(paramlist), nprocs):
    #print "rank", my_rank, "rs=%f Nk=%d ndim=%d fname=%s" % tuple(paramlist[i])
    HFS.main_(*paramlist[i])
