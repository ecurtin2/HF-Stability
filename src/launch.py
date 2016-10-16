import time
    
from pyfiles.lib import HFS_CythonGenerated as HFS
HFS.set_Nk(15)
HFS.set_rs(1.2)
HFS.set_ndim(2)
HFS.py_calc_params()

import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('white')
#plt.figure()
#HFS.plot_energy()
#plt.show()
#
#plt.figure(figsize=(4,4)) 
#HFS.plot_1stBZ()
#plt.show()
#plt.figure()
#HFS.plot_exc_hist()

import numpy as np


from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
N = HFS.get_Nexc() * 2
lo = LinearOperator((N, N), HFS.py_matvec_prod_3H, dtype=np.float64)
t1 = time.time()
vals, vecs = eigsh(lo, k=5, ncv=10, which='LA')
t2 = time.time()
HFS.py_build_matrix()

t3 = time.time()
mat = HFS.get_full_matrix()
t4 = time.time()
npvals, npvecs = np.linalg.eig(mat)

print "np lowest val = ", np.amin(npvals), " took ", t4-t3, " seconds."

print "lowest eval = ", np.amin(vals), " took ", t2-t1, " seconds."
