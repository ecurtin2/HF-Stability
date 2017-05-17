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
Nklist = [80]
rslist = np.logspace(0, 1.4, 10)
caselist = ['cRHF2cRHF']
use_delta = 'false'
paramlist = np.linspace(1, 10, 9)

jobrank = itertools.cycle([1,2,3])

count = 1
lines = []
for nk in Nklist:
    for rs in rslist:
        for case in caselist:
            for param in paramlist:
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

