#ifndef CPP_STABILITY // header guards 
#define CPP_STABILITY
#ifndef PI
	#define PI 3.14159265358979323846264338327
#endif
#define ARMA_NO_DEBUG
#include "armadillo"
#include <map>
#include <vector>
#include <time.h>

namespace HFS {
//Attributes
double  bzone_length, vol, rs, kf, kmax, fermi_energy;
double  two_e_const, deltaK;
arma::uword Nocc, Nvir, Nexc, N_elec, Nk;
int ndim;
arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
arma::vec inp_test_vec, out_vec1, out_vec2;
arma::mat mattest;
arma::umat occ_states, vir_states, excitations;
arma::vec dav_vals;
arma::mat dav_vecs;
int dav_its;

std::string dav_message;

//Methods
double mvec_test();
void   calc_energy_wrap(bool);
void   calc_exc_energy();
arma::uword get_k_to_idx(double[]);
void get_vir_states_inv();
void   calc_energies(arma::umat&, arma::vec&);
double exchange(arma::umat&, arma::uword);
double two_electron(arma::vec, arma::vec);
double two_electron_check(arma::vec, arma::vec, arma::vec, arma::vec);
double get_1B(arma::uword, arma::uword);
double get_3B(arma::uword, arma::uword);
double get_1A(arma::uword, arma::uword);
double get_3A(arma::uword, arma::uword);
double get_3H(arma::uword, arma::uword);
void to_first_BZ(arma::vec&);
void get_inv_exc_map();
void get_vir_N_to_1_map();
arma::uvec inv_exc_map_test;
void build_mattest();
void matvec_prod_arma();
void matvec_prod_me();
arma::vec matvec_prod_3H(arma::vec);
void davidson_wrapper(arma::uword 
                     ,arma::uword 
                     ,arma::uword 
                     ,arma::uword 
                     ,arma::mat   
                     ,double      
                     ,int         
                     );
arma::uword kb_j_to_t(arma::vec, arma::uword);
arma::vec matvec_prod_3A(arma::vec);
arma::vec matvec_prod_3B(arma::vec);
arma::vec occ_idx_to_k(arma::uword);
arma::vec vir_idx_to_k(arma::uword);
std::vector<arma::uword> k_to_idx(arma::vec);
std::map<std::vector<arma::uword>, arma::uword> inv_exc_map;
std::map<std::vector<arma::uword>, arma::uword> vir_N_to_1_map;
void davidson_algorithm(arma::uword  
		       ,arma::uword 
	   	       ,arma::uword  
		       ,arma::uword  
		       ,arma::uword
	   	       ,arma::mat
		       ,double
	   	       ,double (*matrix)(arma::uword, arma::uword)
                       ,arma::vec (*matvec_product)(arma::vec v)
                       ); 
}

#endif
