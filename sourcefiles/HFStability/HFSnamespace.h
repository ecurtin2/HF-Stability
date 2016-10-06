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
extern arma::vec inp_test_vec, out_vec1, out_vec2;
extern arma::mat mattest;
extern arma::umat occ_states, vir_states, excitations;
extern arma::vec dav_vals;
extern arma::mat dav_vecs;
extern arma::mat states;
extern int dav_its;

extern std::string dav_message;

//Methods
extern void print_params();
extern void calc_occ_states();
extern void calc_occ_energies();
extern void calc_vir_energies();
extern void calc_excitations();
extern bool is_vir(double);
extern void   calc_exc_energy();
extern void   calc_energies(arma::umat&, arma::vec&);
extern double exchange(arma::umat&, arma::uword);
extern double two_electron(arma::vec, arma::vec);
extern double two_electron_check(arma::vec, arma::vec, arma::vec, arma::vec);
extern double calc_1B(arma::uword, arma::uword);
extern double calc_3B(arma::uword, arma::uword);
extern double calc_1A(arma::uword, arma::uword);
extern double calc_3A(arma::uword, arma::uword);
extern double calc_3H(arma::uword, arma::uword);
extern void to_first_BZ(arma::vec&);
extern void calc_params();
extern void calc_inv_exc_map();
extern void calc_vir_N_to_1_map();
extern arma::uvec k_to_index(arma::vec);
extern arma::umat k_to_index(arma::mat);
extern arma::uvec inv_exc_map_test;
extern void build_mattest();
extern void matvec_prod_me();
extern arma::vec matvec_prod_3H(arma::vec);
extern void davidson_wrapper(arma::uword, arma::uword, arma::uword, arma::uword, arma::mat, double, int);
extern arma::uword kb_j_to_t(arma::vec, arma::uword);
extern arma::vec matvec_prod_3A(arma::vec);
extern arma::vec matvec_prod_3B(arma::vec);
extern arma::vec occ_idx_to_k(arma::uword);
extern arma::vec vir_idx_to_k(arma::uword);
extern std::vector<arma::uword> k_to_idx(arma::vec);
extern std::map<std::vector<arma::uword>, arma::uword> inv_exc_map;
extern std::map<std::vector<arma::uword>, arma::uword> vir_N_to_1_map;
extern void davidson_algorithm(arma::uword,arma::uword, arma::uword, arma::uword, arma::uword, arma::mat, double, double (*matrix)(arma::uword, arma::uword), arma::vec (*matvec_product)(arma::vec v));
}

#endif
