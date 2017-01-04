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
    extern double exchange(arma::umat&, arma::uword);
    /**< \brief Calculate the exchange energy for the given state

    */
    extern double twoElectronSafe(arma::vec&, arma::vec&, arma::vec&, arma::vec&);
    inline void toFirstBrillouinZone(arma::vec& k){
        // Translate to first brillioun zone, defined on the
        // interval [-pi/a .. pi/a)
        for (unsigned i = 0; i < NDIM; ++i) {
            if (k[i] < -HFS::kmax - SMALLNUMBER) {
                k[i] += HFS::bzone_length;
            } else if (k[i] > HFS::kmax - SMALLNUMBER) {
                k[i] -= HFS::bzone_length;
            }
        }
    }
    extern double twoElectron(arma::vec& k1, arma::vec& k2);

    inline bool isOccupied(double k){
        return (k < (HFS::kf));
    }
    extern arma::uvec kToIndex(arma::vec& k);
    extern arma::umat kToIndex(arma::mat&);
    extern arma::vec occIndexToK(arma::uword);
    extern arma::vec virIndexToK(arma::uword);
    extern int kroneckerDelta(arma::uword, arma::uword);
    extern std::vector<arma::vec> stToKiKaKjKb(arma::uword s, arma::uword t);
}

#endif // HFS_base_funcs_included
