# HF-Stability
Suite of programs to determine stability of solutions to the Hartree-Fock Equations. 

## overview
Contains theoretical background and figures. Summary is in the heg_equations.pdf. This was built from the included .tex file.
All dependencies are also here. The .bib contains all relevant references I've found so far. Additionally, the .pdf  has a diagrammatic layout of the program. 

## src
Contains all source code. The project is written as a Python module made from wrapping C++ using SWIG. 
Data analysis and plotting are done in python, while high performance computation (the actual computation) is done in C++. The C++ maintains full functionality without any external influence, and can be compiled and run on its own. 

## Presentations
Beamer source and compilations for presentations related to the project, such as group meetings. 

## Images
Images generated for the project, which may be used in presentations as well as papers. 
