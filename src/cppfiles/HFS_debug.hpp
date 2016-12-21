#ifndef HFS_debug_included
#define HFS_debug_included

#include <assert.h>
#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"
#include "HFS_matrix_utils.hpp"
#include "HFS_davidson.hpp"

namespace HFS{
    extern double full_diag_min;
    extern bool davidson_agrees_fulldiag();
    extern bool mv_is_working();
    extern double Mv_time, Mv_time2;
    extern double full_diag_time;
    extern bool everything_works();
    extern arma::mat build_matrix(double (*Matrix_func)(arma::uword, arma::uword), arma::uword N);
    extern void time_mv();
    extern void time_mv2();
}

#endif // HFS_debug_included
