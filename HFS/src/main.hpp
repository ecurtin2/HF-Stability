/** @file main.hpp
@author Evan Curtin
@version Revision 0.1
@brief Prototype Main function for Hartree-Fock stability.
@details For the release version, the parameters are
taken as command line input.
@date Wednesday, 04 Jan, 2017
*/

#include <iostream>
#include <cmath>
#include <assert.h>
#include "parameters.hpp"
#include "base_funcs.hpp"
#include "calc_parameters.hpp"
#include "matrix_utils.hpp"
#include "debug.hpp"
#include "fileIO.hpp"

int main(int argc, char* argv[]);
/**<
The main function, called with input parameters from the command line in the release build.
The parameters must be given in the correct order. In the debug and profile builds, the
parameters are set to reasonable values for profiling and debugging.

@param HFS::rs (double) The Wigner-Seitz radius.
@param HFS::Nk (unsigned) The number of k-points per dimension.
@param HFS::OutputFileName (std::string) Name to be used for the output file.
@param HFS::Dav_tol (double) Tolerance for the convergence of the residue norm of the Jacobi-Davidson Algorithm.
@param HFS::Dav_maxits (unsigned) Maximum number of iterations of the Jacobi-Davidson Algorithm.
@param HFS::Dav_maxsubsize (unsigned) Maximum dimension of the subspace before restart of the Jacobi-Davidson Algorithm.
@param HFS::num_guess_evecs (unsigned) Number of initial guess eigenvectors for the Jacobi-Davidson Algorithm
@param HFS::Dav_blocksize (unsigned) Number of trial vectors to add each iteration of the Jacobi-Davidson Algorithm. Must be <= HFS::num_guess_evecs.
@param HFS::Dav_Num_evals (unsigned) Number of extremum eigenvalues to determine.
@param HFS::mycase (std::string) The nature of the Hartre-Fock instability. Currently supported: "cRHF2cUHF" or "cUHF2cUHF".
*/

