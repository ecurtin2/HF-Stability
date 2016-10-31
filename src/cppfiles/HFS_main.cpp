#include "HFS_main.hpp"
#include <chrono>
#include <ctime>
#include <stdio.h>
#include "SLEPcWrapper.hpp"

int main(){
    double rs = 1.2;
    unsigned Nk = 10;
    unsigned ndim = 2;
    unsigned num_guess_evecs = 15;
    unsigned Dav_blocksize = 7;
    unsigned Dav_Num_evals = 5;
    unsigned Dav_maxits = 30;
    unsigned Dav_minits = 10;
    unsigned Maxsubsize = 4000;
    double tolerance = 1E-5;
    std::string outputfilename="test.log";
    return main_(rs, Nk, ndim, num_guess_evecs, Dav_blocksize, Dav_Num_evals, Dav_minits, Dav_maxits, Maxsubsize, tolerance, outputfilename);
}

// The two mains thing is because of SWIG, so I can call main_() from python
int main_(double rs
         ,unsigned Nk
         ,unsigned ndim
         ,unsigned num_guess_vecs
         ,unsigned dav_blocksize
         ,unsigned num_evals
         ,unsigned minits
         ,unsigned maxits
         ,unsigned maxsubsize
         ,double tol
         ,std::string outputfilename)
{

    // Start the timers
    std::chrono::time_point<std::chrono::system_clock> thetime;
    thetime = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(thetime);
    HFS::Computation_Starttime = std::string(std::ctime(&end_time));
    arma::wall_clock timer;
    timer.tic();

    // Get Params
    HFS::rs = rs;
    HFS::Nk = Nk;
    HFS::ndim = ndim;
    HFS::OutputFileName = outputfilename;
    HFS::calc_params();

    HFS::Dav_tol = tol;
    HFS::Dav_minits = minits;
    HFS::Dav_maxits = maxits;
    HFS::Dav_maxsubsize = maxsubsize;
    HFS::num_guess_evecs = num_guess_vecs;
    HFS::Dav_blocksize = dav_blocksize;
    HFS::Dav_Num_evals = num_evals;


/*
    HFS::build_guess_evecs(HFS::num_guess_evecs);
    HFS::davidson_wrapper(2*HFS::Nexc
                          ,HFS::guess_evecs
                          ,HFS::Dav_blocksize
                          ,0
                          ,HFS::Dav_Num_evals
                          ,HFS::Dav_minits
                          ,HFS::Dav_maxits
                          ,HFS::Dav_maxsubsize
                          ,HFS::Dav_tol);


    if (Nk < 31) {
        HFS::davidson_agrees_fulldiag();
    }
    HFS::Total_Calculation_Time = timer.toc();
*/
    HFS::time_mv();
    std::cout << "0" << std::endl;

   // const char* fname = HFS::OutputFileName.c_str();
   // freopen(fname, "w", stdout);
   // HFS::write_output(true);

    std::cout << "1" << std::endl;
    SLEPc::EpS myeps(2*HFS::Nexc, HFS::void_matvec_prod_3H);
    std::cout << "2" << std::endl;
    myeps.SetDimensions(HFS::Dav_Num_evals, HFS::Dav_maxsubsize);
    std::cout << "3" << std::endl;
    myeps.SetTol(HFS::Dav_tol);
    std::cout << "4" << std::endl;
    myeps.SetBlockSize(HFS::Dav_blocksize);
    std::cout << "5" << std::endl;


    std::vector< std::vector<double> > vecs(HFS::num_guess_evecs, std::vector<double>(2*HFS::Nexc, 0.0));
    std::cout << "6" << std::endl;
    for (unsigned i = 0; i < HFS::num_guess_evecs; ++i) {
        vecs[i][i] = 1.0;
    }
    std::cout << "7" << std::endl;

    myeps.SetInitialSpace(vecs);
    std::cout << "8" << std::endl;
    myeps.Solve();
    std::cout << "9" << std::endl;
    myeps.PrintEvals();
    std::cout << "10" << std::endl;
    printf("# of Values Requested: %i\n", myeps.Nevals);
    printf("# of Values Converged: %i\n", myeps.nconv);
    printf("Blocksize: %i\n", myeps.BlockSize);
    printf("Number of guesses: %i\n", myeps.nguess);
    printf("Tolerance: %8.3E\n", myeps.tol);
    myeps.clean();

    fclose(stdout);
    return 0;
}
