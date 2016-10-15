#ifndef HFS_debug_included
#define HFS_debug_included

#include <assert.h>
#include "HFS_params.h"
#include "HFS_base_funcs.h"
#include "HFS_params_calc.h"
#include "HFS_matrix_utils.h"
#include "HFS_davidson.h"

namespace HFS{
    extern arma::mat full_matrix;

    extern bool davidson_agrees_fulldiag();
    extern bool mv_is_working();
    extern bool everything_works();
    extern void build_matrix();
}

#endif // HFS_debug_included
