/** @file parameters.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including extern declarations for global parameters.
@details Definitions are in parameters.cpp
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_PARAMS_INCLUDED
#define HFS_PARAMS_INCLUDED

#define __STDCPP_WANT_MATH_SPEC_FUNCS__

#include "armadillo"
#include "base_funcs.hpp"

/**
    \brief Managing the physical parameters of the calculation
*/

class PhysicalParams {

public:
    PhysicalParams(scalar inp_rs, scalar inp_kf, std::string inp_mycase);

private:
    std::string mycase;                /**< String describing which instability is being found, "cRHF2cUHF", etc */
    scalar bzone_length;               /**< The length of the entire Brillouin zone, = 2*pi / a */
    scalar vol;                        /**< The volume of a unit cell in the direct lattics */
    scalar rs;                         /**< The wigner-seitz radius */
    scalar kf;                         /**< The fermi wave vector */
    scalar kmax;                       /**< The cutoff wavevector */
    scalar fermi_energy;               /**< The energy level of the highest occupied state */
    scalar cond_number;                /**< The (lower limits of) condition number of the matrix being diagonalized */
    scalar two_e_const;                /**< A pre-calculated number used in the two electron integrals */
    scalar deltaK;                     /**< Spacing of the k-points in the reciprocal lattice */
    uint Nocc;                         /**< Number of occupied orbitals */
    uint Nvir;                         /**< Number of virtual orbitals */
    uint Nexc;                         /**< Number of excitations. Not necessarily Nocc*Nvir due to utilizing symmetry */
    uint N_elec;                       /**< Number of electrons (assumes 2 per occupied state, needs to be modified for non-RHF */
    uint Nk;                           /**< Number of k-points in the first brillouin zone. */
    uint ground_state_degeneracy;      /**< Number of excitations with energy within SMALLNUMBER of lowest. */
    arma::vec occ_energies;            /**< Vector containing the energies of occupied states. */
    arma::vec vir_energies;            /**< Vector containing energies of virtual states. */
    arma::vec exc_energies;            /**< Vector containing energy differences between occupied and virtual states. */
    arma::vec kgrid;                   /**< Vector containing the k values of the grid. */
    arma::umat occ_states;             /**< Matrix where the i'th row contains the indices for kgrid of the i'th occupied state. */
    arma::umat vir_states;             /**< Matrix where the i'th row contains the indices for kgrid of the i'th virtual state. */
    arma::umat excitations;            /**< Matrix where the i'th row contains the indices for the corresponding [occupied, virtual] states. */
    # if NDIM == 1
      arma::uvec vir_N_to_1_mat;
    # endif
    # if NDIM == 2
      arma::umat vir_N_to_1_mat;   /**< Matrix/Cube where the value is the virtual state index */
    # endif
    # if NDIM == 3
      arma::ucube vir_N_to_1_mat;
    #endif
    arma::umat inv_exc_mat;          /**< The [i,a]'th element is s, where s labels the excitation i -> a.  */

    scalar calcKf();
    /**< \brief calculate the fermi momentum, Kf.

    Currently allows for ndim = 1, 2 or 3.

    @see kf
    @see rs
    */

    void calcVolAndTwoEConst();
    /**< \brief Calculate the volume and constant used in the two-electron integral.

    @param [in] N_elec
    @param [in] rs
    @param [out] Vol
    @param [out] TwoEConst
    @see vol
    @see N_elec
    @see rs
    @see two_e_const
    */

    void calcStates();
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

    void calcEnergies();
    /**< \brief Calculate the energies of states from the x, y and z indices

    @param inp_states A matrix where each row corresponds to the x, y and z indices
    of each state. Reminder that the momentum is HFS::kgrid[index].
    @param energy_vec reference to a vector where the energies will be stored.
    */

    void calcExcitations();
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

    void calcExcitationEnergies();
    /**< \brief Calculates and sorts the "excitation energies." Sorts excitations

    The "excitation energy" in this program is the difference in energy
    between an occupied and virtual state. The energies are sorted in ascending
    order. HFS::excitations is sorted accordingly. Sets the following variables:
    @see exc_energies
    @see excitations
    */

    void calcLowestEnergyExcitationDegeneracy();
    /**< \brief Calculate the number of states with the lowest energy.

    Calculates how many excitation are within SMALLNUMBER of the lowest
    energy excitation.
    @see SMALLNUMBER
    @see exc_energies
    */

    void calcVirNTo1Map();
    /**< \brief Create the Map to convert between 3/2-index and 1-index representation of virtual state.

    The NDIM compiler directory determines if this is an arma::ucube (3d) or arma::umat
    (2d) or arma::uvec (1d).
    @see NDIM
    @see vir_N_to_1_mat
    @see vir_states
    */

    void calcInverseExcitationMap();
    /**< \brief Create a map that converts from i, a --> s

    The one index denoting the excitation, s corresponds to the excitation from
    occupied state i to virtual state a.
    @see HFS::inv_exc_mat
    */
};

#endif // HFS_PARAMS_INCLUDED
