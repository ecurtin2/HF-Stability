/** @file calc_parameters.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including extern prototypes to calculate system parameters.
@details The definitions are in calc_parameters.cpp.
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_params_calc_included
#define HFS_params_calc_included

#include "parameters.hpp"
#include "base_funcs.hpp"

namespace HFS {
    extern void calcParameters();
    extern void calcKf();
    extern void calcVolAndTwoEConst();
    extern void calcOccupiedStates();
    extern void calcOccupiedEnergies();
    extern void calcVirtualEnergies();
    extern void calcEnergies(arma::umat&, arma::vec&);
    extern void calcExcitations();
    extern void calcExcitationEnergies();
    extern void calcLowestEnergyExcitationDegeneracy();
    extern void calcVirNTo1Map();
    extern void calcInverseExcitationMap();
}
#endif // HFS_params_calc_included
