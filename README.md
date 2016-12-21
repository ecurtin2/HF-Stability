# HF-Stability
Suite of programs to determine stability of solutions to the Hartree-Fock Equations. This repo was made for the solution
of the stability equations for the 2 and 3-dimensional Homogeneous Electron Gas (HEG). 

## /info/
Contains theoretical background and figures. Summary is in the heg_equations.pdf. This was built from the included .tex file.
All dependencies are also here. [This Mendeley group](https://www.mendeley.com/groups/9962001/hfs/papers/) contains all relevant references I've found so far. Additionally, the .pdf  has a diagrammatic layout of the program. 

## /src/
Contains all source code. Data analysis and plotting are done in python, while high performance computation (the actual computation) is done in C++. The C++ maintains full functionality without any external influence, and can be compiled and run on its own, from the command line. I also have a code::blocks project file in the source directory which may be used if desired

The project is built using make. It has 3 build options: profile, debug and release. To build with -o3 optimizations and no debugging symbols, 
```

cd HF-Stability/src 
make Release

```
With optimizations and debugging symbols,
```

cd HF-Stability/src 
make Profile

```
and with no optimizations, while using debugging symbols
```

cd HF-Stability/src 
make Debug

```
The easiest way to run the program is using **launch.py**. Simply edit the *paramlist* to be a list of lists, each containing parameters for one run.
```

cd HF-Stability/src 
python launch.py

```
Since the inclusion of SLEPc to the project, mpi4py no longer functions. Getting SLEPc to work in parallel is a work in progress. This will likely be needed for the 3D case, which is also a work in progress. 


## /presentations/
Beamer source and compilations for presentations related to the project, such as group meetings. 

## /images/
Images generated for the project, which may be used in presentations as well as papers. 
