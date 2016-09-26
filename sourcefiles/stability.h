#ifndef CPP_STABILITY // header guards 
#define CPP_STABILITY
#ifndef PI
	#define PI 3.14159265358979323846264338327
#endif
#include "armadillo"
#include <map>
#include <vector>

namespace HFStability {
	class HEG {
	public:
		//Attributes
		double  bzone_length, vol, rs, kf, kmax, fermi_energy;
                double  two_e_const, deltaK;
		uint64_t    Nocc, Nvir, Nexc, N_elec, Nk;
                int ndim;
		arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
		arma::umat occ_states, vir_states, excitations;

		//Methods
                arma::vec mat_vec_prod(arma::vec);
                void   calc_energy_wrap(bool);
                void   calc_exc_energy();
                arma::uword get_k_to_idx(double[]);
                void get_vir_states_inv();
                void   calc_energies(arma::umat&, arma::vec&);
                double exchange(arma::umat&, arma::uword);
                double two_electron(arma::vec, arma::vec);
                void get_inv_exc_map();
                arma::uvec inv_exc_map_test;
        private:
                std::vector<arma::uword> k_to_idx(arma::vec);
                std::map<std::vector<arma::uword>, arma::uword> inv_exc_map;
                std::map<std::vector<arma::uword>, arma::uword> vir_N_to_1_map;
		double davidson_algorithm(uint64_t, 
				uint64_t,
			   	uint64_t, 
				uint64_t, 
				arma::uword,
			   	arma::mat,
				double, 
			   	double (HFStability::HEG::*)(uint64_t, uint64_t));
	};
}

#endif
