#include "HFS_main.hpp"
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

    HFS::rs              = std::stof(argv[1]);
    HFS::Nk              = std::stoi(argv[2]);
    HFS::ndim            = std::stoi(argv[3]);
    HFS::OutputFileName  = argv[4];
    HFS::Dav_tol         = std::stof(argv[5]);
    HFS::Dav_maxits      = std::stoi(argv[6]);
    HFS::Dav_maxsubsize  = std::stoi(argv[7]);
    HFS::num_guess_evecs = std::stoi(argv[8]);
    HFS::Dav_blocksize   = std::stoi(argv[9]);
    HFS::Dav_Num_evals   = std::stoi(argv[10]);

    HFS::calc_params();

    HFS::time_mv();

    SLEPc::EpS myeps(2*HFS::Nexc, HFS::void_matvec_prod_3H);
    myeps.SetDimensions(HFS::Dav_Num_evals, HFS::Dav_maxsubsize);
    myeps.SetTol(HFS::Dav_tol, HFS::Dav_maxits);
    myeps.SetBlockSize(HFS::Dav_blocksize);

    if (HFS::Nk < 31) {
        HFS::davidson_agrees_fulldiag();
    }


    //std::vector< std::vector<double> > vecs(HFS::num_guess_evecs, std::vector<double>(2*HFS::Nexc, 0.0));
    //for (unsigned i = 0; i < HFS::num_guess_evecs; ++i) {
    //    vecs[i][i] = 1.0;
    //}

    // Try this, weight by how close diags are
    std::vector< std::vector<double> > vecs(HFS::num_guess_evecs, std::vector<double>(2*HFS::Nexc, 0.0));
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
    const char* fname = HFS::OutputFileName.c_str();
    freopen(fname, "w", stdout);
    HFS::write_output(true);

    myeps.clean();
    fclose(stdout);
    return 0;
}
