/** @file main.cpp
@author Evan Curtin
@version Revision 0.1
@brief Main function for Hartree-Fock stability.
@details For the release version, the parameters are
taken as command line input.
@date Wednesday, 04 Jan, 2017
*/

/**
@mainpage
This is the Documentation page generated by Doxygen for the Hartree-
Fock stability of the Homogeneous Electron Gas.
*/


/*
This program is incomplete in the following ways
1. Does not play well with MPI.
2. Does not have a functioning 3D version.
*/


#include "main.hpp"
#include <chrono>
#include <ctime>
#include <stdio.h>
#include "SLEPcWrapper.hpp"

int main(int argc, char* argv[]){
    // Start the timers
    std::chrono::time_point<std::chrono::system_clock> thetime;
    thetime = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(thetime);
    HFS::Computation_Starttime = std::string(std::ctime(&end_time));
    arma::wall_clock timer;
    timer.tic();


    #ifdef RELEASE
        if (argc != 4) {
            std::cout << "Error! Wrong number of arguments" << std::endl;
            exit(EXIT_FAILURE);
        }
        HFS::rs              = std::stof(argv[1]);
        HFS::Nk              = std::stoi(argv[2]);
        HFS::mycase          = argv[3];
    #else
        HFS::rs              = 1.2;
        HFS::Nk              = 15;
        HFS::OutputFileName  = "test.log";
        HFS::mycase          = "cRHF2cUHF";
    #endif // Release

    HFS::Dav_tol         = 1e-6;
    HFS::Dav_maxits      = 30;
    HFS::Dav_maxsubsize  = 1500;
    HFS::num_guess_evecs = 1;
    HFS::Dav_blocksize   = 1;
    HFS::Dav_Num_evals   = 1;

    HFS::calcParameters();
    HFS::setMatrixPropertiesFromCase(); // RHF-UHF etc instability, matrix dimension
    HFS::timeMatrixVectorProduct();



    SLEPc::EpS myeps(HFS::Nmat, HFS::MatVecProduct_func);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    myeps.nprocs = nprocs;
    myeps.SetDimensions(HFS::Dav_Num_evals, HFS::Dav_maxsubsize);
    myeps.SetTol(HFS::Dav_tol, HFS::Dav_maxits);
    myeps.SetBlockSize(HFS::Dav_blocksize);

    // Try this, weight by how close diags are
    std::vector< std::vector<double> > vecs(HFS::num_guess_evecs, std::vector<double>(HFS::Nmat, 0.0));
    for (unsigned i = 0; i < HFS::num_guess_evecs; ++i) {
        arma::vec guessvec;
        arma::vec temp = arma::abs(HFS::exc_energies[i] - HFS::exc_energies) + 1;
        guessvec = (1.0 / temp);
        guessvec /= arma::norm(guessvec);
        vecs[i] = arma::conv_to< std::vector<double> >::from(guessvec);
    }



    myeps.SetInitialSpace(vecs);
    arma::wall_clock davtimer;
    davtimer.tic();
    myeps.Solve();
    HFS::Dav_time = davtimer.toc();
    HFS::Total_Calculation_Time = timer.toc();
    arma::vec temp(myeps.rVals);
    HFS::dav_vals = temp;
    HFS::dav_its = myeps.niter;
    HFS::Dav_nconv = myeps.nconv;
    HFS::cond_number = HFS::exc_energies(HFS::exc_energies.n_elem-1) / HFS::exc_energies(0);
    HFS::Dav_final_val = HFS::dav_vals.min();
//    const char* fname = HFS::OutputFileName.c_str();
//    std::cout << "Calculation finished, writing output to: "<< fname << std::endl;
//    FILE* myfile = freopen(fname, "w", stdout);
    HFS::writeOutput(true);

    myeps.clean();
    fclose(stdout);
    std::cout << "hi" << std::endl;

    #ifndef NDEBUG
        if (HFS::Nmat < 1500) {
            HFS::davidsonAgreesWithFullDiag();
        }

        if ( !HFS::everything_works() ) {
            exit(EXIT_FAILURE);
        }
    #endif //NDEBUG


    return 0;
}
