#include "HFS_main.hpp"
#include <chrono>
#include <ctime>
#include <stdio.h>

int main(){
    double rs = 1.2;
    int Nk = 10;
    int ndim = 2;
    int num_guess_evecs = 10;
    int Dav_blocksize = 10;
    int Dav_Num_evals = 5;
    std::string outputfilename="test.log";
    return main_(rs, Nk, ndim, num_guess_evecs, Dav_blocksize, Dav_Num_evals, outputfilename);
}

// The two mains thing is because of SWIG, so I can call main_() from python
int main_(double rs, int Nk, int ndim, int num_guess_vecs, int dav_blocksize, int num_evals, std::string outputfilename)
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

    if (Nk < 30) {
        HFS::davidson_agrees_fulldiag();
    }

    HFS::num_guess_evecs = num_guess_vecs;
    HFS::Dav_blocksize = dav_blocksize;
    HFS::Dav_Num_evals = num_evals;

    HFS::build_guess_evecs(HFS::num_guess_evecs);
    HFS::davidson_wrapper(2*HFS::Nexc, HFS::guess_evecs, HFS::Dav_blocksize,
                          0,                // which
                          HFS::Dav_Num_evals);

    HFS::Total_Calculation_Time = timer.toc();

    const char* fname = HFS::OutputFileName.c_str();
    freopen(fname, "w", stdout);
    HFS::write_output(true);
    fclose(stdout);
    return 0;
}
