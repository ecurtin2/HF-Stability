#ifndef CPP_STABILITY // header guards
#define CPP_STABILITY
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

namespace HFS {
//Attributes
extern double  bzone_length, vol, rs, kf, kmax, fermi_energy;
extern double  two_e_const, deltaK;
extern arma::uword Nocc, Nvir, Nexc, N_elec, Nk;
extern int ndim;
extern arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
extern arma::umat occ_states, vir_states, excitations;
extern arma::umat vir_N_to_1_mat, inv_exc_mat;

//Davidson stuff
extern std::string dav_message;
extern arma::vec dav_vals;
extern arma::mat dav_vecs;
extern int dav_its;

//Checks/Debug
extern arma::mat full_matrix;

//Methods


// Parameter calculation
extern void calc_params();
extern void calc_kf();
extern void calc_vol_and_two_e_const();
extern void calc_occ_states();
extern void calc_occ_energies();
extern void calc_vir_energies();
extern void calc_energies(arma::umat&, arma::vec&);
extern void calc_excitations();
extern void calc_exc_energy();
extern void calc_vir_N_to_1_mat();
extern void calc_inv_exc_mat();

// Common functions needed multiple places
extern double exchange(arma::umat&, arma::uword);
extern double two_electron(arma::vec&, arma::vec&);
extern double two_electron_check(arma::vec&, arma::vec&, arma::vec&, arma::vec&); // checks momentum conserve, used in calc_matrix(i,j)
extern void to_first_BZ(arma::vec&);
extern bool is_vir(double);
extern arma::uvec k_to_index(arma::vec&);
extern arma::umat k_to_index(arma::mat&);
extern arma::vec occ_idx_to_k(arma::uword);
extern arma::vec vir_idx_to_k(arma::uword);

// Matrix and Matrix-vector products
extern arma::vec matvec_prod_3A(arma::vec&);
extern arma::vec matvec_prod_3B(arma::vec&);
extern arma::vec matvec_prod_3H(arma::vec&);
extern double calc_1B(arma::uword, arma::uword);
extern double calc_3B(arma::uword, arma::uword);
extern double calc_1A(arma::uword, arma::uword);
extern double calc_3A(arma::uword, arma::uword);
extern double calc_3H(arma::uword, arma::uword);
extern arma::uword kb_j_to_t(arma::vec&, arma::uword);  // only used in matvec_prod_3A & 3B

//Davidson Algorithm
extern void davidson_wrapper(arma::uword N, arma::mat guess_evecs=arma::eye(2*HFS::Nexc,1), arma::uword block_size=1, int which=0,arma::uword num_of_roots=1, arma::uword max_its=20, arma::uword max_sub_size=HFS::Nocc, double tolerance=10E-8);
extern void davidson_algorithm(arma::uword,arma::uword, arma::uword, arma::uword, arma::uword, arma::mat, double, double (*matrix)(arma::uword, arma::uword), arma::vec (*matvec_product)(arma::vec& v));

// Testing/Debugging Functions
extern bool davidson_agrees_fulldiag();
extern bool mv_is_working(double tol=SMALLNUMBER);
extern bool everything_works();
extern void build_matrix();

// Output Control
extern void print_params();
}

#endif
