/** @file fileIO.hpp
@author Evan Curtin
@version Revision 0.1
@brief Methods for writing output file.
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_fileIO_included
#define HFS_fileIO_included

#include "parameters.hpp"
#include "base_funcs.hpp"
#include "calc_parameters.hpp"
#include "matrix_utils.hpp"
#include "debug.hpp"

namespace HFS{
    extern void writeOutput(bool detail=false);
    /** \brief Writes the output to a file, with or without vectors
     *
     * \param detail True to print vectors (much larger file).
     *
     */

    std::string centerString(std::string s, int width);
    /** \brief Return the string, padded by spaces to center it.
     *
     * \param s The string to be centered.
     * \param width The width of the centering window (# of characters)
     * \return The centered string.
     */

}

#endif // HFS_fileIO_included
