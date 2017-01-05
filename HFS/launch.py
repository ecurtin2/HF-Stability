import numpy as np
import tempfile
import sys
import os

def get_fname(rs, Nk, ndim):
    pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
    ext = '.log'
    outdir = 'log'
    fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)
    return fname

Nk = 12
#rsrange = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5]
rsrange = [1.2]
ndim = 2
nguess = 1
blocksize = 1
num_evals = 1
maxits = 50
maxsubsize = 1000
tolerance = 1e-6
mycase = "cRHF2cUHF"

paramlist = [['./HFS', rs, Nk, get_fname(rs, Nk, ndim), tolerance, maxits,
              maxsubsize, nguess, blocksize, num_evals, mycase] for rs in rsrange]

for i in range(len(paramlist)):
     print 'Starting Job: ',  ' '.join([str(j) for j in paramlist[i]])
     os.system('mpirun -np 2 ' + ' '.join([str(j) for j in paramlist[i]]))
