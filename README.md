# HF-Stability
Suite of programs to determine stability of solutions to the Hartree-Fock Equations. 

##overview
Contains theoretical background and figures. Summary is in the heg_equations.pdf. This was built from the included .tex file.
All dependencies are also here. The .bib contains all relevant references I've found so far. Additionally, the .pdf  has a diagrammatic layout of the program. 

##source
Contains all source code. The project is written as a Python-wrapped C++ class using Cython. 
The class methods that need speed are written in C++ while the rest are written in python in the .pyx, which is compiled by Cython. These components may possibly be made faster by using static typing but this will probably be negligible compared to the bottleneck, which is the matrix-vector multiplications in Davidson's algorithm. 
