#ifndef HFS_davidson_included
#define HFS_davidson_included

#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"
#include "HFS_matrix_utils.hpp"

namespace HFS{
    extern arma::mat guess_evecs;
    extern std::string Davidson_Stopping_Criteria;
    extern arma::vec dav_vals, dav_lowest_vals;
    extern arma::mat dav_vecs;
    extern int dav_its;
    extern int num_guess_evecs;
    extern int Dav_blocksize;
    extern int Dav_Num_evals;
    extern double Dav_time;
    extern void build_guess_evecs (int N, int which=0);
    extern void davidson_wrapper(arma::uword N, arma::mat guess_evecs, arma::uword block_size=1, int which=0, arma::uword num_of_roots=1, arma::uword max_its=20, arma::uword max_sub_size=1000, double tolerance=10E-8);
    extern void davidson_algorithm(arma::uword,arma::uword, arma::uword, arma::uword, arma::uword, arma::mat&, double, double (*matrix)(arma::uword, arma::uword), arma::vec (*matvec_product)(arma::vec& v));
}

#endif // HFS_davidson_included
