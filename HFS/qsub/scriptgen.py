import numpy as np
import tempfile
import os

NDIM = 3
CASE = "cRHF2cUHF"


f = open("templateqsub.sh", "r")
content = f.readlines()
f.close()

def getCmd(rs, Nk, ndim, case):
    cmd = '${exe} ' + str(rs) + ' ' + str(Nk) + ' ' + case + ' > ${outfile}\n'
    return cmd

def getFilename(rs, Nk, ndim):
    pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
    ext = ''
    outdir = '.'
    fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)
    return fname

def writeFile(rs, Nk, ndim, case, fname):
    newcontent = []
    for line in content:
        if "OUTFILE" in line:
            newcontent += "outfile="+ fname.replace("./", "") + ".log\n"
        elif "EXECUTABLE" in line:
            newcontent += getCmd(rs, Nk, ndim, case)
        else:
            newcontent += line
    scriptname = fname.replace("./", "qsub-") + ".sh"
    f = open(scriptname, "w+")
    for line in newcontent:
        f.write(line)
    f.close()
    return scriptname

qsublist = []
rslist = [1.2]
Nklist = range(14, 40)
for rs in rslist:
    for Nk in Nklist:
        fname = getFilename(rs, Nk, NDIM)
        scriptname = writeFile(rs, Nk, NDIM, CASE, fname)
        os.system("qsub " + scriptname)
