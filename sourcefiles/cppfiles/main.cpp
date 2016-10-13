#include "HFSnamespace.h"


int main()
{
    clock_t t, t2;
    HFS::rs = 1.2;
    HFS::Nk = 25;
    HFS::ndim = 2;
    HFS::calc_params();

    arma::mat guess = arma::eye<arma::mat>(2*HFS::Nexc, 60);
    t = clock();
    HFS::davidson_wrapper(2*HFS::Nexc, guess, 10, 0, 1, 50, 2*HFS::Nexc);
    t2 = clock() - t;
    std::cout << "Dav took " << ((float)t2) / CLOCKS_PER_SEC << " seconds" << std::endl;
    HFS::print_params();
    HFS::everything_works();
}
