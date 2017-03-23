#!/bin/bash
#$ -S /bin/bash            # use bash shell
NODE
#$ -pe orte 16             # parallel environment: use N cores
#$ -cwd                    # execute from current working directory
#$ -e error.out
#$ -o output.out

scr=/scratch/ecurtin2
outdir=/home/ecurtin2/git/HF-Stability/analysis/log
DEFINE_EXECUTABLE

#This line changed by scriptgen.py
OUTFILE

# Build Scratch Directory
mkdir -p ${scr}

# Change to scratch directory
cd ${scr}

# Execute program
#This line changed by scriptgen.py
CALL_EXECUTABLE

# Return to original directory
cd -

# Copy from scr, outfile defined after scriptgen.py replacements
mv ${scr}/${outfile} ${outdir}

