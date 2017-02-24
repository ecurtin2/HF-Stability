/** @file parameters.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including extern declarations for global parameters.
@details Definitions are in parameters.cpp
@date Wednesday, 04 Jan, 2017
*/

#ifndef HFS_params_included
#define HFS_params_included

#ifndef PI
    #define PI 3.14159265358979323846264338327
#endif
#ifndef SMALLNUMBER
    #define SMALLNUMBER 1E-12
#endif
#include "armadillo"

/** \namespace HFS
    \brief Global parameters. Used mostly for traceability of parameters.
*/

namespace HFS{
    extern double bzone_length;               /**< The length of the entire Brillouin zone, = 2*pi / a */
    extern double vol;                        /**< The volume of a unit cell in the direct lattics */
    extern double rs;                         /**< The wigner-seitz radius */
    extern double kf;                         /**< The fermi wave vector */
    extern double kmax;                       /**< The cutoff wavevector */
    extern double fermi_energy;               /**< The energy level of the highest occupied state */
    extern double cond_number;                /**< The (lower limits of) condition number of the matrix being diagonalized */
    extern double two_e_const;                /**< A pre-calculated number used in the two electron integrals */
    extern double deltaK;                     /**< Spacing of the k-points in the reciprocal lattice */
    extern double Total_Calculation_Time;     /**< Time from main() start to finish */
    extern std::string Computation_Starttime; /**< Time of starting main() (date, time, year) */
    extern std::string OutputFileName;        /**< Name of the file to be written to */
    extern std::string mycase;                /**< String describing which instability is being found, "cRHF2cUHF", etc */
    extern arma::uword Nocc;                  /**< Number of occupied orbitals */
    extern arma::uword Nvir;                  /**< Number of virtual orbitals */
    extern arma::uword Nexc;                  /**< Number of excitations. Not necessarily Nocc*Nvir due to symmetry */
    extern arma::uword N_elec;                /**< Number of electrons (assumes 2 per occupied state, needs to be modified for non-RHF */
    extern arma::uword Nmat;                  /**< Size of stability matrix is Nmat x Nmat. */
    extern unsigned Nk;                       /**< Number of k-points in the first brillouin zone. */
    extern unsigned ground_state_degeneracy;  /**< Number of excitations with energy within SMALLNUMBER of lowest. */
    extern arma::vec occ_energies;            /**< Vector containing the energies of occupied states. */
    extern arma::vec vir_energies;            /**< Vector containing energies of virtual states. */
    extern arma::vec exc_energies;            /**< Vector containing energy differences between occupied and virtual states. */
    extern arma::vec kgrid;                   /**< Vector containing the k values of the grid. */
    extern arma::umat occ_states;             /**< Matrix where the i'th row contains the indices for kgrid of the i'th occupied state. */
    extern arma::umat vir_states;             /**< Matrix where the i'th row contains the indices for kgrid of the i'th virtual state. */
    extern arma::umat excitations;            /**< Matrix where the i'th row contains the indices for the corresponding [occupied, virtual] states. */
    #if NDIM == 2
      extern   arma::umat vir_N_to_1_mat;       /**< Matrix/Cube where the value is the virtual state index */
    #elif NDIM == 3
      extern   arma::ucube vir_N_to_1_mat;
    #endif // NDIM;
    extern arma::umat inv_exc_mat;            /**< The [i,a]'th element is s, where s labels the excitation i -> a.  */
    extern void (*MatVecProduct_func)(arma::vec& v, arma::vec& Mv); /**< Function pointer for the matrix-vector product. Is set by HFS::setMatrixPropertiesFromCase */
    extern double (*Matrix_func)(arma::uword i, arma::uword j);     /**< Function pointer for the matrix elements. Is set by HFS::setMatrixPropertiesFromCase */
    extern unsigned dav_its;                  /**< Number of iterations to converge in Davidson's Algorithm  */
    extern arma::vec dav_vals;                /**< All eigenvalues returned by the last iteration of Davidson's Algorithm. */
    extern unsigned num_guess_evecs;          /**< Number of eigenvectors used as initial guess for Davidson's Algorithm. */
    extern unsigned Dav_blocksize;            /**< Block size Davidson's Algorithm. */
    extern unsigned Dav_Num_evals;            /**< Number of eigenvalues requested for Davidson's Algorithm. */
    extern unsigned Dav_nconv;                /**< Number of converged eigenpairs returned by Davidson's Algorithm. */
    extern double Dav_tol;                    /**< Tolerance for the residual norm for Davidson's Algorithm. */
    extern double Dav_final_val;              /**< Lowest eigenvalue returned by the last iteration of Davidson's Algorithm. */
    extern unsigned Dav_maxits;               /**< Maximum number of iterations for Davidson's Algorithm. */
    extern unsigned Dav_minits;               /**< Minimum number of iterations for Davidson's Algorithm. */
    extern unsigned Dav_maxsubsize;           /**< Maximum size of the subspace before restart for Davidson's Algorithm. */
    extern double Dav_time;                   /**< Time taken until convergence for Davidson's Algorithm. */
}
#endif // HFS_params_included
