#!/bin/bash
#$ -S /bin/bash            # use bash shell
#$ -q all.q@compute-0-0,all.q@compute-0-1,all.q@compute-0-2,all.q@compute-0-4,all.q@compute-0-5,all.q@compute-0-6,all.q@compute-0-7   # Queue jobs on any node  
#$ -pe orte 1              # parallel environment: use N cores
#$ -cwd                    # execute from current working directory
#$ -e error.out
#$ -o output.out

scr=/scratch/ecurtin2
outdir=/home/ecurtin2/git/HF-Stability/HFS/log
exe=/home/ecurtin2/git/HF-Stability/HFS/HFS 

#This line changed by scriptgen.py
outfile=000001.200_00037_3_bNQmFT.log

# Build Scratch Directory
mkdir -p ${scr}

# Change to scratch directory
cd ${scr}

# Execute program
#This line changed by scriptgen.py
${exe} 1.2 37 cRHF2cUHF > ${outfile}

# Return to original directory
cd -

# Copy from scr, outfile defined after scriptgen.py replacements
mv ${scr}/${outfile} ${outdir}

