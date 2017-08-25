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

#ifndef PI
    #define PI 3.14159265358979323846264338327
#endif
#ifndef SMALLNUMBER
    #define SMALLNUMBER 1E-12
#endif

#include "armadillo"

typedef unsigned uint;
typedef double   scalar;

/** \namespace HFS
    \brief Global parameters. Used mostly for traceability of parameters.
*/

namespace HFS{
    extern scalar bzone_length;               /**< The length of the entire Brillouin zone, = 2*pi / a */
    extern scalar twoE_parameter_1dCase;            /**< Constant for the delta function interaction version of 1D. */
    extern bool use_delta_1D;                 /**< Flag to determine if delta function or exponential integral for 1d. */
    extern scalar vol;                        /**< The volume of a unit cell in the direct lattics */
    extern scalar rs;                         /**< The wigner-seitz radius */
    extern scalar kf;                         /**< The fermi wave vector */
    extern scalar kmax;                       /**< The cutoff wavevector */
    extern scalar fermi_energy;               /**< The energy level of the highest occupied state */
    extern scalar cond_number;                /**< The (lower limits of) condition number of the matrix being diagonalized */
    extern scalar two_e_const;                /**< A pre-calculated number used in the two electron integrals */
    extern scalar deltaK;                     /**< Spacing of the k-points in the reciprocal lattice */
    extern scalar Total_Calculation_Time;     /**< Time from main() start to finish */
    extern std::string computation_started; /**< Time of starting main() (date, time, year) */
    extern std::string OutputFileName;        /**< Name of the file to be written to */
    extern std::string mycase;                /**< String describing which instability is being found, "cRHF2cUHF", etc */
    extern uint Nocc;                  /**< Number of occupied orbitals */
    extern uint Nvir;                  /**< Number of virtual orbitals */
    extern uint Nexc;                  /**< Number of excitations. Not necessarily Nocc*Nvir due to symmetry */
    extern uint N_elec;                /**< Number of electrons (assumes 2 per occupied state, needs to be modified for non-RHF */
    extern uint Nmat;                  /**< Size of stability matrix is Nmat x Nmat. */
    extern uint Nk;                       /**< Number of k-points in the first brillouin zone. */
    extern uint ground_state_degeneracy;  /**< Number of excitations with energy within SMALLNUMBER of lowest. */
    extern arma::vec occ_energies;            /**< Vector containing the energies of occupied states. */
    extern arma::vec vir_energies;            /**< Vector containing energies of virtual states. */
    extern arma::vec exc_energies;            /**< Vector containing energy differences between occupied and virtual states. */
    extern arma::vec kgrid;                   /**< Vector containing the k values of the grid. */
    extern arma::umat occ_states;             /**< Matrix where the i'th row contains the indices for kgrid of the i'th occupied state. */
    extern arma::umat vir_states;             /**< Matrix where the i'th row contains the indices for kgrid of the i'th virtual state. */
    extern arma::umat excitations;            /**< Matrix where the i'th row contains the indices for the corresponding [occupied, virtual] states. */
    # if NDIM == 1
      extern arma::uvec vir_N_to_1_mat;
    # endif
    # if NDIM == 2
      extern   arma::umat vir_N_to_1_mat;       /**< Matrix/Cube where the value is the virtual state index */
    # endif
    # if NDIM == 3
      extern   arma::ucube vir_N_to_1_mat;
    #endif
    extern arma::umat inv_exc_mat;            /**< The [i,a]'th element is s, where s labels the excitation i -> a.  */
    extern void (*MatVecProduct_func)(arma::vec& v, arma::vec& Mv); /**< Function pointer for the matrix-vector product. Is set by HFS::setMatrixPropertiesFromCase */
    extern arma::mat (*Matrix_generator)();   /**< Function pointer for the matrix elements. Is set by HFS::setMatrixPropertiesFromCase */
    extern uint dav_its;                  /**< Number of iterations to converge in Davidson's Algorithm  */
    extern arma::vec dav_vals;                /**< All eigenvalues returned by the last iteration of Davidson's Algorithm. */
    extern uint num_guess_evecs;          /**< Number of eigenvectors used as initial guess for Davidson's Algorithm. */
    extern arma::vec exact_evals;         /**< Only used in debug mode, holds the full eigenvalue spectra of the matrix from HFS::Matrix_generator */
    extern uint dav_blocksize;            /**< Block size Davidson's Algorithm. */
    extern uint dav_num_evals;            /**< Number of eigenvalues requested for Davidson's Algorithm. */
    extern uint dav_nconv;                /**< Number of converged eigenpairs returned by Davidson's Algorithm. */
    extern scalar dav_tol;                    /**< Tolerance for the residual norm for Davidson's Algorithm. */
    extern scalar dav_min_eval;              /**< Lowest eigenvalue returned by the last iteration of Davidson's Algorithm. */
    extern uint dav_maxits;               /**< Maximum number of iterations for Davidson's Algorithm. */
    extern uint Dav_minits;               /**< Minimum number of iterations for Davidson's Algorithm. */
    extern uint dav_max_subsize;           /**< Maximum size of the subspace before restart for Davidson's Algorithm. */
    extern scalar dav_time;                   /**< Time taken until convergence for Davidson's Algorithm. */
    extern int N_MV_PROD;
    extern std::vector<double> mv_times;
    extern scalar dav_singlet_a_plus_b, dav_singlet_a_minus_b;
    extern scalar dav_triplet_a_plus_b, dav_triplet_a_minus_b;
}
#endif // HFS_PARAMS_INCLUDED
