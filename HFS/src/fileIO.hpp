/** @file fileIO.hpp
@author Evan Curtin
@version Revision 0.1
@brief Methods for writing output file.
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_FILEIO_INCLUDED
#define HFS_FILEIO_INCLUDED

#include <string>
#include "armadillo"

template <class T>
std::string ToJSON(std::ofstream& output, const arma::Col<T>& v, const std::string& varname);
template <class T>
std::string ToJSON(std::ofstream& output, const arma::Mat<T>& M, const std::string& varname);

#endif // HFS_FILEIO_INCLUDED
