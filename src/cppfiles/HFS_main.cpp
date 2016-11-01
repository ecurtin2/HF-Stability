#include "HFS_main.hpp"
#include <chrono>
#include <ctime>
#include <stdio.h>
#include "SLEPcWrapper.hpp"

int main(){
    // Start the timers
    std::chrono::time_point<std::chrono::system_clock> thetime;
    thetime = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(thetime);
    HFS::Computation_Starttime = std::string(std::ctime(&end_time));
    arma::wall_clock timer;
    timer.tic();

    // Get Params
    std::cin >> HFS::rs;
    std::cin >> HFS::Nk;
    std::cin >> HFS::ndim;
    std::cin >> HFS::OutputFileName;
    std::cin >> HFS::Dav_tol;
    std::cin >> HFS::Dav_maxits;
    std::cin >> HFS::Dav_maxsubsize;
    std::cin >> HFS::num_guess_evecs;
    std::cin >> HFS::Dav_blocksize;
    std::cin >> HFS::Dav_Num_evals;
    HFS::calc_params();
    // Get Params
//    HFS::rs = 1.2;
//    HFS::Nk = 15;
//    HFS::ndim = 2;
//    HFS::OutputFileName = "test.log";
//    HFS::calc_params();
//    HFS::Dav_tol = 1e-5;
//    HFS::Dav_maxits = 30;
//    HFS::Dav_maxsubsize = 1000;
//    HFS::num_guess_evecs = 15;
//    HFS::Dav_blocksize = 7;
//    HFS::Dav_Num_evals = 5;

    if (HFS::Nk < 31) {
        HFS::davidson_agrees_fulldiag();
    }
    HFS::Total_Calculation_Time = timer.toc();
    HFS::time_mv();

    SLEPc::EpS myeps(2*HFS::Nexc, HFS::void_matvec_prod_3H);
    myeps.SetDimensions(HFS::Dav_Num_evals, HFS::Dav_maxsubsize);
    myeps.SetTol(HFS::Dav_tol, HFS::Dav_maxits);
    myeps.SetBlockSize(HFS::Dav_blocksize);

    std::vector< std::vector<double> > vecs(HFS::num_guess_evecs, std::vector<double>(2*HFS::Nexc, 0.0));
    for (unsigned i = 0; i < HFS::num_guess_evecs; ++i) {
        vecs[i][i] = 1.0;
    }

    myeps.SetInitialSpace(vecs);
    arma::wall_clock davtimer;
    davtimer.tic();
    myeps.Solve();
    HFS::Dav_time = davtimer.toc();

    arma::vec temp(myeps.rVals);
    HFS::dav_vals = temp;
    HFS::dav_its = myeps.niter;
    HFS::Dav_nconv = myeps.nconv;
    HFS::cond_number = HFS::exc_energies(HFS::exc_energies.n_elem-1) / HFS::exc_energies(0);
    HFS::Dav_final_val = HFS::dav_vals.min();
    const char* fname = HFS::OutputFileName.c_str();
    freopen(fname, "w", stdout);
    HFS::write_output(true);

    myeps.clean();
    fclose(stdout);
    return 0;
}
