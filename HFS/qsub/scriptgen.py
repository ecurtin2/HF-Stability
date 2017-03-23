import numpy as np
import itertools
import tempfile
import os

NDIM = 3
caselist = ['cUHF2cUHF', 'cRHF2cUHF']
rslists = {
	   'cUHF2cUHF' : [1.2]
	  ,'cRHF2cUHF' : [0.8, 1.0, 1.2, 2.0, 4.0, 5.0] 
          }
Nklists = {
	   'cUHF2cUHF' : [33]
	  ,'cRHF2cUHF' : range(13, 49, 4)
          }
Nodes = itertools.cycle([4, 5, 6, 7]) # iterator that loops through these variables
			              # goes back to start after reaching end. 

f = open("templateqsub.sh", "r")
content = f.readlines()
f.close()

def getCmd(rs, Nk, ndim, case):
    cmd = '${exe} --rs ' + str(rs) + ' --Nk ' + str(Nk) + ' --mycase ' + case + ' --fname ${outfile}\n'
    return cmd

def getFilename(rs, Nk, ndim, case):
    pre = '%010.3f_%05d_%1d_'%(rs, Nk, ndim)
    pre += case + '_'
    ext = ''
    outdir = '.'
    fname = tempfile.mktemp(suffix=ext, prefix=pre, dir=outdir)
    return fname

def writeFile(rs, Nk, ndim, case, fname):
    newcontent = []
    for line in content:
        if "OUTFILE" in line:
            newcontent += "outfile="+ fname.replace("./", "") + ".json\n"
        elif "NODE" in line:
	    newcontent += '#$ -q all.q@compute-0-'+ str(next(Nodes)) + ' # Queue jobs on this nodei\n'
        elif "CALL_EXECUTABLE" in line:
            newcontent += getCmd(rs, Nk, ndim, case)
        elif "DEFINE_EXECUTABLE" in line:
            newcontent += 'exe=/home/ecurtin2/git/HF-Stability/HFS/bin/HFSrelease'+ str(NDIM) + 'D'
        else:
            newcontent += line
    scriptname = fname.replace("./", "qsub-") + ".sh"
    f = open(scriptname, "w+")
    for line in newcontent:
        f.write(line)
    f.close()
    return scriptname

qsublist = []
for case in caselist:
    for rs in rslists[case]:
        for Nk in Nklists[case]:
            fname = getFilename(rs, Nk, NDIM, case)
            scriptname = writeFile(rs, Nk, NDIM, case, fname)
            os.system("qsub " + scriptname)
