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
		long    Nocc, Nvir, Nexc, N_elec, ndim, Nk;
		arma::mat  states;
		arma::vec  energies;
		arma::umat excitations;
		arma::uvec occ_states, vir_states;

		//Methods
        double min_eigval(long, long, long, long, long, long, bool, double, double*);
		//double energy(long long unsigned int);
		double two_electron_3d(double[], double[], double[]);
		double two_electron_2d(double[], double[], double[]);
                void calc_energies_2d();
                double exchange_2d(arma::uword);
                double tw_electron_2d(arma::uword, arma::uword);
                void calc_energies_3d();
                double exchange_3d(arma::uword);
                double tw_electron_3d(arma::uword, arma::uword);
		double davidson_algorithm(long, 
				long,
			   	long, 
				long, 
				arma::uword,
			   	arma::mat,
				double, 
			   	double (HFStability::HEG::*)(long, long));
	};
}

#endif
