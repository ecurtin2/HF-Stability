/*
#ifndef PI
    #define PI 3.14159265358979323846264338327
#endif
#ifndef SMALLNUMBER
    #define SMALLNUMBER 10E-10
#endif
//#define ARMA_NO_DEBUG
#include "armadillo"
#include <map>
#include <vector>
#include <time.h>
*/
#include "HFSnamespace.h"

namespace HFS {
//Attributes
double  bzone_length, vol, rs, kf, kmax, fermi_energy;
double  two_e_const, deltaK;
arma::uword Nocc, Nvir, Nexc, N_elec, Nk;
int ndim;
arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
arma::umat occ_states, vir_states, excitations;
arma::umat vir_N_to_1_mat, inv_exc_mat;

//Davidson stuff
arma::mat guess_evecs;
std::string dav_message;
arma::vec dav_vals;
arma::mat dav_vecs;
int dav_its;

//Checks/Debug
arma::mat full_matrix;

//Methods
// Parameter calculation
void calc_params();
void calc_kf();
void calc_vol_and_two_e_const();
void calc_occ_states();
void calc_occ_energies();
void calc_vir_energies();
void calc_energies(arma::umat&, arma::vec&);
void calc_excitations();
void calc_exc_energy();
void calc_vir_N_to_1_mat();
void calc_inv_exc_mat();

// Common functions needed multiple places
double exchange(arma::umat&, arma::uword);
double two_electron(arma::vec&, arma::vec&);
double two_electron_check(arma::vec&, arma::vec&, arma::vec&, arma::vec&); // checks momentum conserve, used in calc_matrix(i,j)
void to_first_BZ(arma::vec&);
bool is_vir(double);
arma::uvec k_to_index(arma::vec&);
arma::umat k_to_index(arma::mat&);
arma::vec occ_idx_to_k(arma::uword);
arma::vec vir_idx_to_k(arma::uword);

// Matrix and Matrix-vector products
arma::vec matvec_prod_3A(arma::vec&);
arma::vec matvec_prod_3B(arma::vec&);
arma::vec matvec_prod_3H(arma::vec&);
double calc_1B(arma::uword, arma::uword);
double calc_3B(arma::uword, arma::uword);
double calc_1A(arma::uword, arma::uword);
double calc_3A(arma::uword, arma::uword);
double calc_3H(arma::uword, arma::uword);
arma::uword kb_j_to_t(arma::vec&, arma::uword);  // only used in matvec_prod_3A & 3B

//Davidson Algorithm
void build_guess_evecs (int N, int which);
void davidson_wrapper(arma::uword N, arma::mat guess_evecs, arma::uword block_size, int which, arma::uword num_of_roots, arma::uword max_its, arma::uword max_sub_size, double tolerance);
void davidson_algorithm(arma::uword,arma::uword, arma::uword, arma::uword, arma::uword, arma::mat&, double, double (*matrix)(arma::uword, arma::uword), arma::vec (*matvec_product)(arma::vec& v));

// Testing/Debugging Functions
bool davidson_agrees_fulldiag();
bool mv_is_working();
bool everything_works();
void build_matrix();

//Output Control
void print_params();
}
//#endif
