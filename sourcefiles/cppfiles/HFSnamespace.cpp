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
#include "HFSnamespace.h"

namespace HFS {
//Attributes
double  bzone_length, vol, rs, kf, kmax, fermi_energy;
double  two_e_const, deltaK;
arma::uword Nocc, Nvir, Nexc, N_elec, Nk;
int ndim;
arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
arma::vec inp_test_vec, out_vec1, out_vec2;
arma::mat full_matrix;
arma::umat occ_states, vir_states, excitations;
arma::vec dav_vals;
arma::mat dav_vecs;
arma::mat states;
arma::umat vir_N_to_1_mat;
int dav_its;

std::string dav_message;

//Methods
void calc_vir_N_to_1_mat();
void calc_kf();
void calc_vol_and_two_e_const();
void print_params();
void calc_occ_states();
void calc_occ_energies();
void calc_vir_energies();
void calc_excitations();
bool is_vir(double);
void   calc_energy_wrap(bool);
void   calc_exc_energy();
//arma::uword get_k_to_idx(double[]);
void   calc_energies(arma::umat&, arma::vec&);
double exchange(arma::umat&, arma::uword);
double two_electron(arma::vec, arma::vec);
double two_electron_check(arma::vec, arma::vec, arma::vec, arma::vec);
double calc_1B(arma::uword, arma::uword);
double calc_3B(arma::uword, arma::uword);
double calc_1A(arma::uword, arma::uword);
double calc_3A(arma::uword, arma::uword);
double calc_3H(arma::uword, arma::uword);
void to_first_BZ(arma::vec&);
void calc_params();
void calc_inv_exc_map();
void calc_vir_N_to_1_map();
void calc_inv_exc_mat();
arma::umat inv_exc_mat;
arma::uvec k_to_index(arma::vec&);
arma::umat k_to_index(arma::mat);
arma::uvec inv_exc_map_test;
void build_matrix();
void matvec_prod_arma();
arma::vec matvec_prod_3H(arma::vec);
void davidson_wrapper(arma::uword N, arma::mat guess_evecs=arma::eye<arma::mat>(2*HFS::Nexc,1), arma::uword block_size=1, int which=0, arma::uword num_of_roots=1, arma::uword max_its=20, arma::uword max_sub_size=HFS::Nocc, double tolerance=10E-5);
bool davidson_agrees_fulldiag();
bool mv_is_working(double tol=SMALLNUMBER);
bool everything_works();


arma::uword kb_j_to_t(arma::vec&, arma::uword);
arma::vec matvec_prod_3A(arma::vec);
arma::vec matvec_prod_3B(arma::vec);
arma::vec occ_idx_to_k(arma::uword);
arma::vec vir_idx_to_k(arma::uword);
std::vector<arma::uword> k_to_idx(arma::vec);
std::map<std::vector<arma::uword>, arma::uword> inv_exc_map;
std::map<std::vector<arma::uword>, arma::uword> vir_N_to_1_map;
void davidson_algorithm(arma::uword, arma::uword, arma::uword, arma::uword, arma::uword, arma::mat, double, double (*matrix)(arma::uword, arma::uword), arma::vec (*matvec_product)(arma::vec v));
}

#endif
