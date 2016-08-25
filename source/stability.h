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
		arma::mat  states;
		arma::vec  energies, exc_energies, kgrid, occ_energies, vir_energies;
		arma::umat occ_states_idx, vir_states_idx, excitations;
                arma::uvec occ2state, vir2state;
		arma::mat occ_states, vir_states;
                uint64_t my_test;

		//Methods
        double min_eigval(long, long, long, long, long, long, bool, double, double*);
                //2d
                double two_electron_2d(double[], double[]);
                //void calc_energies_2d_idx();
                void calc_occ_energies_2d_idx();
                void calc_vir_energies_2d_idx();
                double exchange_occ_2d_idx(arma::uword);
                double exchange_vir_2d_idx(arma::uword);
                void calc_energies_2d(arma::umat&, arma::vec&);
                double exchange_2d(arma::umat&, arma::uword);
                //3d
                void calc_energies_3d(arma::umat&, arma::vec&);
                double exchange_3d(arma::umat&, arma::uword);
                double two_electron_3d(double[], double[]);
                void calc_energies_3d_wrap(bool);
                //void calc_energies_3d_idx();
               // double exchange_3d_idx(arma::uword);
               // double two_electron_3d_idx(double[], double[]);

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
