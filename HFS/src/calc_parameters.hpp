/** @file calc_parameters.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including extern prototypes to calculate system parameters.
@details The definitions are in calc_parameters.cpp.
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_PARAMS_CALC_INCLUDED
#define HFS_PARAMS_CALC_INCLUDED

#include "parameters.hpp"
#include "base_funcs.hpp"

namespace HFS {
    extern void calcParameters();
    /**< \brief Wrapper for all parameter calculations.

    @see HFS::calcKf()
    @see HFS::kmax
    @see HFS::bzone_length
    @see HFS::fermi_energy
    @see HFS::kgrid
    @see HFS::deltaK
    @see HFS::calcOccupiedStates();
    @see HFS::calcVolAndTwoEConst();
    @see HFS::calcOccupiedEnergies();
    @see HFS::calcVirtualEnergies();
    @see HFS::calcExcitations();
    @see HFS::calcExcitationEnergies();
    @see HFS::calcLowestEnergyExcitationDegeneracy();
    @see HFS::calcVirNTo1Map();
    @see HFS::calcInverseExcitationMap();
    */

    extern scalar calcKf(scalar rs, uint ndim);
    /**< \brief calculate the fermi momentum, Kf.

    Currently allows for ndim = 1, 2 or 3.

    @see HFS::kf
    @see HFS::rs
    @see PI
    */

    extern void calcVolAndTwoEConst(uint N_elec, scalar rs, scalar& Vol, scalar& TwoEConst);
    /**< \brief Calculate the volume and constant used in the two-electron integral.

    @param [in] N_elec
    @param [in] rs
    @param [out] Vol
    @param [out] TwoEConst
    @see HFS::vol
    @see HFS::N_elec
    @see HFS::rs
    @see HFS::two_e_const
    */

    extern void calcStates (arma::vec& kgrid
                           ,uint Nk
                           ,uint ndim
                           ,uint& Nocc
                           ,uint& Nvir
                           ,uint& N_elec
                           ,arma::umat& occ_states
                           ,arma::umat& vir_states
                           );
    /**< \brief Calculate the occupied & virtual states, their number and the number of electrons.

    @param [in] kgrid The gridpoints per dimension in k-space.
    @param [in] Nk The number of kpoints per dimension.
    @param [in] ndim The number of dimensions.
    @param [out] Nocc The number of occupied states.
    @param [out] Nvir The number of virtual states.
    @param [out] N_elec The number of electrons.
    @param [out] occ_states The occupied states indices.
    @param [out] vir_states The virtual state indices.
    */

    extern void calcEnergies(arma::umat& inp_states, arma::vec& energy_vec);
    /**< \brief Calculate the energies of states from the x, y and z indices

    @param inp_states A matrix where each row corresponds to the x, y and z indices
    of each state. Reminder that the momentum is HFS::kgrid[index].
    @param energy_vec reference to a vector where the energies will be stored.
    */

    extern void calcExcitations(arma::vec& kgrid
                        ,uint Nk
                        ,scalar deltaK
                        ,uint Nocc
                        ,uint Nvir
                        ,arma::umat& occ_states
                        ,arma::umat& vir_states
                        ,arma::umat& excitations
                        ,arma::vec&  exc_energies
                        ,uint& Nexc);
    /**< \brief Determine excitations in the x direction.

    @param [in] kgrid grid of kpoints.
    @param [in] Nk Number of k points per dimension
    @param [in] Nocc Number of occupied states
    @param [in] Nvir Number of virtual states
    @param [in] occ_states occupied states
    @param [in] vir_states virtual states
    @param [out] excitations Matrix where each row contains [occupied_uint, virtual_uint]
    @param [out] exc_energies Energy difference of occ and vir state in excitation
    @param [out] Nexc Number of excitations
    */

    extern void calcExcitationEnergies();
    /**< \brief Calculates and sorts the "excitation energies." Sorts excitations

    The "excitation energy" in this program is the difference in energy
    between an occupied and virtual state. The energies are sorted in ascending
    order. HFS::excitations is sorted accordingly. Sets the following variables:
    @see exc_energies
    @see excitations
    */

    extern void calcLowestEnergyExcitationDegeneracy();
    /**< \brief Calculate the number of states with the lowest energy.

    Calculates how many excitation are within SMALLNUMBER of the lowest
    energy excitation.
    @see SMALLNUMBER
    @see exc_energies
    */

    extern void calcVirNTo1Map();
    /**< \brief Create the Map to convert between 3/2-index and 1-index representation of virtual state.

    The NDIM compiler directory determines if this is an arma::ucube (3d) or arma::umat
    (2d).
    @see NDIM
    @see vir_N_to_1_mat
    @see vir_states
    */

    extern void calcInverseExcitationMap();
    /**< \brief Create a map that converts from i, a --> s

    The one index denoting the excitation, s corresponds to the excitation from
    occupied state i to virtual state a.
    @see HFS::inv_exc_mat
    */
}
#endif // HFS_params_calc_included
