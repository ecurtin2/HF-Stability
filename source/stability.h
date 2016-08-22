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
		double  bzone_length, vol, rs, kf, fermi_energy;
                double  two_e_const;
		uint64_t    Nocc, Nvir, Nexc, N_elec, ndim, Nk;
		arma::mat  states;
		arma::vec  energies;
		arma::umat excitations;
		arma::mat occ_states, vir_states;
                uint64_t my_test;

		//Methods
        double min_eigval(long, long, long, long, long, long, bool, double, double*);
		//double energy(long long unsigned int);
		double two_electron_3d(double[], double[], double[]);
		double two_electron_2d(double[], double[], double[]);
                void calc_energies_2d();
                double exchange_2d(arma::uword);
<<<<<<< HEAD
                double two_electron_2d(double[], double[]);
                void calc_energies_3d();
                double exchange_3d(arma::uword);
                double two_electron_3d(double[], double[]);
=======
                double tw_electron_2d(arma::uword, arma::uword);
                void calc_energies_3d();
                double exchange_3d(arma::uword);
                double tw_electron_3d(arma::uword, arma::uword);
>>>>>>> parent of 7a522d9... Fixed 3D, altered cyarma to function more reliably.
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
