#ifndef HFS_davidson_included
#define HFS_davidson_included

#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"
#include "HFS_matrix_utils.hpp"

namespace HFS{
    extern arma::mat guess_evecs;
    extern std::string Davidson_Stopping_Criteria;
    extern arma::vec dav_lowest_vals, dav_vals;
    extern arma::mat dav_vecs;
    extern unsigned dav_its;
    extern unsigned num_guess_evecs;
    extern unsigned Dav_blocksize;
    extern unsigned Dav_Num_evals;
    extern unsigned Dav_nconv;
    extern double Dav_tol;
    extern double Dav_final_val;
    extern unsigned Dav_maxits;
    extern unsigned Dav_minits;
    extern unsigned Dav_maxsubsize;
    extern double Dav_time;
    extern arma::vec dav_iteration_timer;
    extern void build_guess_evecs (int N, int which=0);

    extern void mod_gram_schmidt(arma::vec& v, arma::mat& V);
    extern void davidson_wrapper(arma::uword N
                         ,arma::mat   guess_evecs
                         ,unsigned  block_size
                         ,unsigned  which
                         ,unsigned  num_of_roots
                         ,unsigned  min_its
                         ,unsigned  max_its
                         ,unsigned  max_sub_size
                         ,double    tolerance
                         );

    extern void davidson_algorithm(arma::uword N
                           ,unsigned min_its
                           ,unsigned max_its
                           ,unsigned max_sub_size
                           ,unsigned num_of_roots
                           ,unsigned block_size
                           ,arma::mat&  guess_evecs
                           ,double      tolerance
                           ,double      (*matrix)(arma::uword, arma::uword)
                           ,arma::vec   (*matvec_product)(arma::vec& v)
                           );



}

#endif // HFS_davidson_included
