import itertools
import tempfile
import numpy as np

def getFilename(rs, Nk, ndim, case):
    pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
    pre += case + '_'
    ext = ''
    outdir = '.'
    fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)
    return fname

exe = './bin/HFSrelease1D'
Nklist = [100, 200, 300]
rslist = [0.001, 0.01, 0.1, 0.5, 1.0]
caselist = ['cRHF2cUHF', 'cRHF2cRHF']
use_deltas = ['true']
paramlist = np.linspace(1, 10, 4)

jobrank = itertools.cycle([1, 2])

count = 1
lines = []
for nk in Nklist:
    for rs in rslist:
        for case in caselist:
            for param in paramlist:
                for use_delta in use_deltas:
                    cmd = exe + ' --Nk ' + str(nk) + ' --rs ' + str(rs) + ' --mycase ' + case
                    cmd += ' --use_delta_1D ' + use_delta + ' --twoE_parameter_1dCase ' + str(param)
                    cmd += ' --fname ' + getFilename(rs, nk, 1, case) + '.json &\n'
                    lines.append(cmd)
                    if next(jobrank) == 2:
                        lines.append('echo ----------------Running set \#' + str(count) + ' -----------------------------\n')
                        count += 1
                        lines.append('wait\n')

with open('launch.sh', 'w') as f:
    for line in lines:
        f.write(line)

