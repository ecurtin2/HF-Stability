/** @file base_funcs.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including extern prototypes for low-level functions used throughout the codebase.
@details The definitions are in base_funcs.cpp.
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_base_funcs_included
#define HFS_base_funcs_included

#include "parameters.hpp"

namespace HFS{
    extern double exchange(arma::umat& states, arma::uword i);
    /**< \brief Calculate the exchange energy for the given state.
    @param states Either occ_states or vir_states. Will determine the exchange
    energy for the i'th occupied or the i'th virtual state depending on input.
    @param i The index of the state to be considered.
    @return the exchange contribution to the energy
    */

    extern double twoElectronSafe(arma::vec& k1, arma::vec& k2, arma::vec& k3, arma::vec& k4);
    /**< \brief Calculate the two electron integral, <k1 k2|1/r|k3 k4>

    The two electron integral is defined as,
    \f[
    \left< k_1 k_2 \left| \frac{1}{r_{12}} \right| k_3 k_4\right>
    \f].
    Any combination of 4 states is acceptable. The conservation of momentum is checked
    for each combination, and 0 is returned in the even that momentum is not conserved.
    @param k1, k2, k3, k4 Vectors containing the kx, ky, ... for the 4 input states.
    @return The two electron integral
    */

    inline void toFirstBrillouinZone(arma::vec& k){
    /** \brief Translate vector in-place to first Brillioun zone.

    Defined on the interval [-pi/a .. pi/a). The vector is assumed to be within the first
    or second BZ, therefore is only translated a maximum of HFS::bzone_length in each
    dimension.
    @param k The vector in k-space.
    @see HFS::bzone_length

    */
        for (unsigned i = 0; i < NDIM; ++i) {
            if (k[i] < -HFS::kmax - SMALLNUMBER) {
                k[i] += HFS::bzone_length;
            } else if (k[i] > HFS::kmax - SMALLNUMBER) {
                k[i] -= HFS::bzone_length;
            }
        }
    }

    extern double twoElectron(arma::vec& k1, arma::vec& k3);
    /**< \brief Calculate the two electron integral, assuming momentum conservation.

    The two electron integral is defined as,
    \f[
    \left< k_1 k_2 \left| \frac{1}{r_{12}} \right| k_3 k_4\right>
    \f].
    @param k1, k3 The first state in the bra and ket.
    @return The value of the two electron integral.
    */

    inline bool isOccupied(double k){
    /** \brief true if k < kf, else false.
        @see HFS::kf
     */
        return (k < (HFS::kf));
    }

    //extern arma::uvec kToIndex(arma::vec& k);
    extern void kToIndex(arma::vec& k, arma::uvec& idx);
    /**< \brief Given k-vector, return corresponding vector of indices in each dimension.

    Each element in k is converted to the corresponding index. Evenly spaced
    gridpoints are assumed. The indices are related to momentum via HFS::kgrid; <br>
    k[i] = HFS::kgrid[indices[i]].

    @param k vector in k-space
    @return vector of indices
    @see HFS::kgrid

    */

    extern arma::umat kToIndex(arma::mat& k);
    /**< \brief kToIndex, overloaded for matrix

    @param k matrix of k-points
    @return matrix of indices
    @see kToIndex()
    @see HFS::kgrid
    */

    extern void occIndexToK(arma::uword i, arma::vec& k);
    /**< \brief Return the momentum of the i'th occupied state.

    @param i The index of the occupied state.
    @return vector of the kx, ky, ... momentum of the i'th occupied state.
    */

    extern arma::vec virIndexToK(arma::uword i);
    /**< \brief Return the momentum of the i'th virtual state.

    @param i The index of the virtual state.
    @return vector of the kx, ky, ... momentum of the i'th virtual state.
    */

    extern int kroneckerDelta(arma::uword i, arma::uword j);
    /**< \brief return 1 if i = j, else 0.

    @param i, j indices
    @return 1 if i = j, else 0.
    */

    extern std::vector<arma::vec> stToKiKaKjKb(arma::uword s, arma::uword t);
    /**< \brief Given excitation indices s & t, return corresponding ki, kj, ka, kb.

    S corresponds to occupied i to virtual a.
    T corresponds to occupied j to virtual b.
    @param s, t Excitation labels.
    @return Vector of 4 vectors, {ki, kj, ka, kb}. In this order.
    */
}

#endif // HFS_base_funcs_included
