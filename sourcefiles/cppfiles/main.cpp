#include "HFSnamespace.h"


int main()
{
    HFS::rs = 1.2;
    HFS::Nk = 10;
    HFS::ndim = 2;
    HFS::calc_params();
    arma::wall_clock timer;
    arma::mat guess = arma::eye<arma::mat>(2*HFS::Nexc, 60);

    timer.tic();
    HFS::davidson_wrapper(2*HFS::Nexc, guess, 10, 0, 1, 50, 2*HFS::Nexc);
    double t3 = timer.toc();
    std::cout << "Dav took " << t3 << " seconds" << std::endl;
    HFS::print_params();
    HFS::everything_works();
    return 0;
}
