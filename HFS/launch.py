import numpy as np
import multiprocessing
import subprocess
import tempfile
import time
import os

Nkrange = range(12, 15)
rs = 1.2
ndim = 3
mycase = "cRHF2cUHF"


def get_fname(rs, Nk, ndim):
    pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
    ext = '.log'
    outdir = '.'
    fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)
    return fname

def run(params):
    cmd = './HFS ' + ' '.join([str(j) for j in params])
    cmd += ' > ' + get_fname(params[0], params[1], ndim) 
    cmd+= ' &'
    print 'Starting Job: ', cmd 
    os.system(cmd)

paramlist = [[rs, Nk, mycase] for Nk in Nkrange]

for params in paramlist:
    run(params)


