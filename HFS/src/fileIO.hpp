/** @file fileIO.hpp
@author Evan Curtin
@version Revision 0.1
@brief Methods for writing output file.
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_FILEIO_INCLUDED
#define HFS_FILEIO_INCLUDED

#include "parameters.hpp"
#include "base_funcs.hpp"
#include "calc_parameters.hpp"
#include "matrix_vectorproducts.hpp"
#include "debug.hpp"

namespace HFS{

    extern void writeJSON(std::string fname, bool detail);
    /** \brief Writes the output to a file in JSON format, with or without vectors
     *
     * \param fname  Name of output file (function does not add .json to string).
     * \param detail True to print vectors (much larger file).
     *
     */

    template <class T>
    extern void writeArmaVecToJSON(std::ofstream& output, const arma::Col<T>& v, const std::string& varname);
    template <class T>
    extern void writeArmaMatToJSON(std::ofstream& output, const arma::Mat<T>& M, const std::string& varname);
}

#endif // HFS_FILEIO_INCLUDED
