#include "HFS_main.hpp"
#include <chrono>
#include <ctime>
#include <stdio.h>

int main(){
    double rs = 1.2;
    unsigned Nk = 10;
    unsigned ndim = 2;
    unsigned num_guess_evecs = 60;
    unsigned Dav_blocksize = 20;
    unsigned Dav_Num_evals = 6;
    unsigned Dav_maxits = 30;
    unsigned Dav_minits = 10;
    unsigned Maxsubsize = 4000;
    double tolerance = 1E-10;
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

    HFS::num_guess_evecs = HFS::ground_state_degeneracy * 5;
    HFS::Dav_Num_evals = HFS::ground_state_degeneracy;
    HFS::Dav_blocksize = HFS::ground_state_degeneracy * 2;

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

    HFS::time_mv();

    const char* fname = HFS::OutputFileName.c_str();
    freopen(fname, "w", stdout);
    HFS::write_output(true);
    fclose(stdout);
    return 0;
}
