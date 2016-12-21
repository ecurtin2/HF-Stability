// This is a SWIG wrapper file

%module HFS
%include <std_string.i>
%{
#define SWIG_FILE_WITH_INIT

/* Includes the headers in the wrapper code */
#include "../cppfiles/HFS_main.hpp"
#include "../cppfiles/HFS_params.hpp"
#include "../cppfiles/HFS_base_funcs.hpp"
#include "../cppfiles/HFS_params_calc.hpp"
#include "../cppfiles/HFS_matrix_utils.hpp"
#include "../cppfiles/HFS_davidson.hpp"
#include "../cppfiles/HFS_debug.hpp"
#include "../cppfiles/HFS_fileIO.hpp"
%}

/* Now include ArmaNpy typemaps */
%include "armanpy.i"

/* Some minimal excpetion handling */
%exception {
    try {
        $action
    }
    catch( std::exception & e  ) { PyErr_SetString( PyExc_RuntimeError, e.what() ); SWIG_fail; } 
}

/* Parse the header file to generate wrappers */
%include "../cppfiles/HFS_main.hpp"
%include "../cppfiles/HFS_params.hpp"
%include "../cppfiles/HFS_base_funcs.hpp"
%include "../cppfiles/HFS_params_calc.hpp"
%include "../cppfiles/HFS_matrix_utils.hpp"
%include "../cppfiles/HFS_davidson.hpp"
%include "../cppfiles/HFS_debug.hpp"
%include "../cppfiles/HFS_fileIO.hpp"
