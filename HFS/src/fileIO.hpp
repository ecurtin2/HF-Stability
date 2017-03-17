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
    extern void writeOutput(bool detail=false);
    /** \brief Writes the output to a file, with or without vectors
     *
     * \param detail True to print vectors (much larger file).
     *
     */

    extern void writeJSON(bool detail);

    template <class T>
    void vecToJSON(std::ofstream& output, const arma::Col<T>& v, const std::string& varname);
    template <class T>
    void writeArmaVec(std::ofstream& output, const arma::Col<T>& v);
    template <class T>
    void writeArmaMat(std::ofstream& output, const arma::Mat<T>& M, const std::string& varname);

    extern std::string centerString(std::string s, int width);
    /** \brief Return the string, padded by spaces to center it.
     *
     * \param s The string to be centered.
     * \param width The width of the centering window (# of characters)
     * \return The centered string.
     */

}

#endif // HFS_FILEIO_INCLUDED
