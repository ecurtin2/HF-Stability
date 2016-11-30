#ifndef HFS_debug_included
#define HFS_debug_included

#include <assert.h>
#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"
#include "HFS_matrix_utils.hpp"
#include "HFS_davidson.hpp"

namespace HFS{
    extern arma::mat full_matrix;
    extern double full_diag_min;
    extern bool davidson_agrees_fulldiag();
    extern bool mv_is_working();
    extern double Mv_time, Mv_time2;
    extern double full_diag_time;
    extern bool everything_works();
    extern void build_matrix();
    extern void time_mv();
    extern void time_mv2();
}

#endif // HFS_debug_included
