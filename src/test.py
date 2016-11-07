import numpy as np
import sys
import os



Nklist =  range(10, 100)
rs = 1.2
ndim = 2
nguess = 50
blocksize = 20
neval = 8
maxit = 50
maxsubsize = 1000
tol = 1e-8
fname = 'test.log'

tests = [['./HFS', rs, Nk, ndim, fname, tol, maxit, maxsubsize, nguess, blocksize, neval] for Nk in Nklist]

strtests = [' '.join([str(j) for j in test]) for test in tests]



for test in strtests:
    print test
#os.system(test)
