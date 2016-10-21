#include <iostream>
#include <cmath>
#include <assert.h>
#include "HFS_params.hpp"
#include "HFS_base_funcs.hpp"
#include "HFS_params_calc.hpp"
#include "HFS_matrix_utils.hpp"
#include "HFS_davidson.hpp"
#include "HFS_debug.hpp"
#include "HFS_fileIO.hpp"

int main();
int main_(double rs
         ,unsigned Nk
         ,unsigned ndim
         ,unsigned num_guess_vecs
         ,unsigned dav_blocksize
         ,unsigned num_evals
         ,unsigned minits
         ,unsigned maxits
         ,unsigned maxsubsize
         ,double tol
         ,std::string outputfilename);
