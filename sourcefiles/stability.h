#ifndef CPP_STABILITY // header guards 
#define CPP_STABILITY
#ifndef PI
	#define PI 3.14159265358979323846264338327
#endif
#include "armadillo"

namespace HFStability {
	class HEG {
	public:
		//Attributes
		double  bzone_length, vol, rs, kf, kmax, fermi_energy;
                double  two_e_const, deltaK;
		uint64_t    Nocc, Nvir, Nexc, N_elec, ndim, Nk;
		arma::vec  occ_energies, vir_energies, exc_energies, kgrid;
		arma::umat occ_states, vir_states, excitations;

		//Methods
                arma::vec& mat_vec_prod_2d(arma::vec);
                void   calc_energy_wrap(bool);
                void   calc_exc_energy();
                //2d
                void   calc_energies_2d(arma::umat&, arma::vec&);
                double two_electron_2d(double[], double[]);
                double exchange_2d(arma::umat&, arma::uword);
                arma::uword get_k_to_idx(double[]);
                void get_vir_states_inv_2d();
                //3d
                void   calc_energies_3d(arma::umat&, arma::vec&);
                double exchange_3d(arma::umat&, arma::uword);
                double two_electron_3d(double[], double[]);
        private:

                //independent of dimension
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
