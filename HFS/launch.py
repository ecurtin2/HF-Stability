import tempfile
import os
from mpi4py import MPI
from subprocess import call
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


rs = 1.2
Nk = 12

ndim = 3
mycase = "cRHF2cUHF"

pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
ext = '.log'
outdir = '.'
fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)

cmd = './HFS ' + str(rs) + ' ' + str(Nk) + ' ' + mycase + ' > ' + fname
os.system(cmd)
comm.Barrier()
