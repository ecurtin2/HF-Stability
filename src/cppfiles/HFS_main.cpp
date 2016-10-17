#include "HFS_main.hpp"


int main(){
    return main_();
}

// The two mains thing is because of SWIG, so I can call main_() from python
int main_()
{
    HFS::rs = 1.2;
    HFS::Nk = 11;
    HFS::ndim = 2;
    HFS::calc_params();
    arma::wall_clock timer;
    HFS::build_guess_evecs(60);

    timer.tic();
    HFS::davidson_wrapper(2*HFS::Nexc, HFS::guess_evecs, 10, 0, 1, 50, 2*HFS::Nexc);
    double t = timer.toc();
    std::cout << "Dav took " << t << " seconds" << std::endl;
    HFS::print_params();
    HFS::everything_works();
    return 0;
}
