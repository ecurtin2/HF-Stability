#ifndef HFS_fileIO_included
#define HFS_fileIO_included

#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"
#include "HFS_matrix_utils.hpp"
#include "HFS_davidson.hpp"
#include "HFS_debug.hpp"

namespace HFS{
    extern void write_output(bool detail=false);
    extern std::string centerstring(std::string s, int width);
}

#endif // HFS_fileIO_included
