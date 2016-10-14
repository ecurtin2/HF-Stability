#include <iostream>
#include <cmath>
#include "HFS_params.h"
#include "HFS_base_funcs.h"
#include "HFS_params_calc.h"
#include "HFS_matrix_utils.h"
#include "HFS_davidson.h"
#include "HFS_debug.h"
#include "HFS_fileIO.h"

int main()
{
    HFS::rs = 1.2;
    HFS::Nk = 10;
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
