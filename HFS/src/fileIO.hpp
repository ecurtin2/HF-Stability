#ifndef HFS_fileIO_included
#define HFS_fileIO_included

#include "parameters.hpp"
#include "base_funcs.hpp"
#include "calc_parameters.hpp"
#include "matrix_utils.hpp"
#include "debug.hpp"

namespace HFS{
    extern void writeOutput(bool detail=false);
    std::string centerString(std::string s, int width);
}

#endif // HFS_fileIO_included
