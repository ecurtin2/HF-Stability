//Stability analysis, the state information is passed in from python
#include <iostream>
#include <iomanip>
#include <cmath>
#include <assert.h>
#include <time.h>
#include "stability.h"
#include "armadillo"
#define ARMA_NO_DEBUG


double HFStability::HEG::energy(unsigned int state)
{
	double energy;
	double kin = 0; //kinetic energy
	double k[ndim], kprime[ndim];

	for (int i = 0; i < ndim; ++i) {
		k[i] = states(state, i);
		kin += k[i]*k[i];
	}
	// ||k||/2m
	kin /= 2.0;

	double exc = 0;
	for (int i = 0; i < Nocc; ++i) {
		for (int j = 0; j < ndim; ++j){
			kprime[j] = states(occ_states(i), j);
		}
		//4th is always k by momentum conservation, 2 is occupation #
		exc += 2.0 * two_electron_2d(k, kprime, kprime);
	}
	exc /= bzone_length*bzone_length;
	energy = kin - exc;
	return energy;
}


double HFStability::HEG::two_electron_3d(double kp[], double kq[], double kr[])
{
		//This is momentum conserving
		double k[ndim];
		for (int i = 0; i < ndim; ++i) {
			k[i] = kp[i] + kq[i] - kr[i];
		}

		//Translate into first BZ
		double bound = bzone_length / 2.0;
		for (int i = 0; i < ndim; ++i) {
			if (k[i] < -bound){
				k[i] += bzone_length;
			}else if (k[i] > bound){
				k[i] -= bzone_length;
			}
		}

		const double tolerance = 10E-10;
		double norm = 0.0;
	    for (int i = 0; i < ndim; ++i) {
			norm += (kp[i] - kr[i]) * (kp[i] - kr[i]);
	    }
	
		//avoid singularities	
		if (norm < tolerance){
			return 0.0;
		}
		return 4.0 * PI	/ (vol * norm);
		}

double HFStability::HEG::two_electron_2d(double kp[], double kq[], double kr[])
{
		//This is momentum conserving
		double k[ndim];
		for (int i = 0; i < ndim; ++i) {
			k[i] = kp[i] + kq[i] - kr[i];
		}

		//Translate into first BZ
		double bound = bzone_length / 2.0;
		for (int i = 0; i < ndim; ++i) {
			if (k[i] < -bound){
				k[i] += bzone_length;
			}else if (k[i] > bound){
				k[i] -= bzone_length;
			}
		}

		const double tolerance = 10E-10;
		double norm = 0.0;
	    for (int i = 0; i < ndim; ++i) {
			norm += (kp[i] - kr[i]) * (kp[i] - kr[i]);
	    }
	
		//avoid singularities	
		if (norm < tolerance){
			return 0.0;
		}
		return 2.0 * PI	/ (vol * sqrt(norm));
		}

double HFStability::HEG::davidson_algorithm(long N,
		long max_its, 
		long max_sub_size,
	   	long num_of_roots,
		arma::uword block_size,
		arma::mat(guess_evecs),
		double tolerance, 
		double (HFStability::HEG::*matrix)(long, long))
{

	//Initialize.	
	long sub_size = guess_evecs.n_cols;
	long old_sub_size = 0;	
	arma::uword num_new_vecs = sub_size;
	arma::vec old_evals;
	arma::mat old_evecs = guess_evecs;
	arma::mat mat_vec_prod(N, 0, arma::fill::zeros);
	arma::mat ritz_vecs = guess_evecs;
		

	clock_t t;//debug
	
	arma::mat init_guess(N, num_of_roots);
	for (long i = 0; i < num_of_roots; ++i) {
		init_guess.col(i) = guess_evecs.col(i);
	}

		

	//Iterate the block Davidson algorithm.
	for (long i=0 ; i < max_its ; ++i){
		arma::mat sub_mat(sub_size, sub_size, arma::fill::zeros);
		//std::cout << "i = " << i << std::endl;
	
		// Matrix Vector product starts here
		double sum = 0;
		num_new_vecs = sub_size - old_sub_size;
		arma::mat temp_mat(N, num_new_vecs, arma::fill::zeros);
		arma::mat temp_mat_t(num_new_vecs, N, arma::fill::zeros);
		double tempnum;	
		//guess evecs into row major ordering, this step is fast compared
		//to striding access within the loop
		arma::mat guess_evecs_t = guess_evecs.t();

		//This is matrix-matrix mult M*V, it looks this way to minimize
		//function calls to this->matrix which were a limiting step.
		for (arma::uword j = 0; j < N; ++j) {
			for (arma::uword l = 0; l < N; ++l) {
				tempnum = (this->*matrix)(j,l);
				for (arma::uword k = old_sub_size; k < sub_size; ++k) {
					//guess_evecs_t is the row major version of guess_evecs
					temp_mat(j, k-old_sub_size) += tempnum * guess_evecs_t(k,l);
				}
			}
		}
		// MV complete

		//temp_mat has new matrix vector products
		mat_vec_prod = arma::join_rows(mat_vec_prod, temp_mat);

		// V.t * MV
		sub_mat = guess_evecs_t * mat_vec_prod;

		//Diagonalize subspace matrix.
		arma::vec sub_evals;
		arma::mat sub_evecs;
		arma::eig_sym(sub_evals, sub_evecs, sub_mat);
		arma::mat temp;

		//sort eigenvals, vecs by value 
		arma::mat Q, R;
		arma::qr_econ(Q, R, sub_evecs);
		sub_evecs = Q;
		arma::uvec indices = arma::sort_index(sub_evals);
		sub_evecs = sub_evecs.cols(indices);
		sub_evals = sub_evals.elem(indices);
		
		old_evecs = sub_evecs;
		ritz_vecs = guess_evecs * sub_evecs;
		/*Current implementation is the Diagonally Preconditioned Residue (DPR) method. 
		  As far as I know right now, this is the original method propossed by Davidson
		  Apparently, sometimes people simply call this the Davidson method. */			
		
		//Get the residues and append to subspace if needed.
		//Sub_size changes within this loop
		old_sub_size = sub_size;
		arma::vec norms(block_size, arma::fill::zeros);
		for (long j=0; j < block_size; ++j){
			arma::vec residue(N, arma::fill::zeros);
			double rayleigh_quotient = 0.0;
			double x_k = 0.0;
			//rayleigh quotient is x.T * M * x
			for (long k = 0; k < N; ++k) {
				x_k = ritz_vecs(k,j);	
				for (long l = 0; l < N; ++l) {
					rayleigh_quotient += x_k * (this->*matrix)(k,l) * ritz_vecs(l,j);
				}
			}

			for (long k=0; k < N; ++k) {
				sum = 0.0;
				for (long m=0; m < N; ++m) {	
					if (m == k){
						sum += ((this->*matrix)(k,k) - rayleigh_quotient) * ritz_vecs(m,j);
					}else{
						sum += (this->*matrix)(k,m) * ritz_vecs(m,j);
					}
				residue(k) = sum;
				}
			}
		
			//Append only the residues with norm > tolerance.
			double norm = arma::norm(residue);
			norms(j) = norm;
			if (norm > tolerance) {
				arma::vec correction(N, arma::fill::zeros);

				//If diagonal element = eval, get singularity
				bool apply_correction = true;
				for (long k = 0; k < old_sub_size; ++k) {
					if ( fabs( (this->*matrix)(k,k) - sub_evals(j) ) <= 10E-10) {
						apply_correction = false;
					}
				}

				//This is the DPR correction
				for (long k=0 ; k < N ; ++k){
					correction(k) = ( -1.0/ ( (this->*matrix)(k,k) - rayleigh_quotient ) ) * residue(k);
				}

				//Only add if no problems encountered
				if (apply_correction) {
					correction = correction/arma::norm(correction);
					guess_evecs = arma::join_rows(guess_evecs, correction);
					sub_size += 1;
				}
			}
		}

		//Print to screen
		std::cout << "Eval = " << sub_evals(0) << " Norm = " << norms(0) << std::endl; 
		
		//Error handling and exit conditions
		if (sub_size > max_sub_size) {
			std::cout << "Error in Davidson: Subspace too big!" << std::endl;
			return sub_evals(0);

		}else if (i == max_its - 1) {
			std::cout << "Davidson Error: Maximum # of iterations reached!" << std::endl;
			return sub_evals(0);
	
		}else if (old_sub_size == sub_size) {
			std::cout << "Davidson: Subspace converged in " << i + 1 << " iterations!"  << std::endl;
			return sub_evals(0);

		}else if ( arma::all(norms.rows(0,num_of_roots-1)  < tolerance)) {
			std::cout << "Davidson: All requested norms converged in " << i + 1 << " iterations!"  << std::endl;
			return sub_evals(0);
		}			

		//Make sure the new guess space is orthonormal
		arma::qr_econ(Q, R, guess_evecs);
		//Enforce sums of elements of eigenvector matrix are positive
		//with no loss of generality. 
		for (int i = 0; i < sub_size; ++i) {
			if (arma::sum(Q.col(i)) < 0.0) {
				Q.col(i) = -1.0 * Q.col(i);
			}
		}
		guess_evecs = Q;

	}
	//If we got here there's a problem
	//NOTE THIS SHOULD BE HANDLED MORE ELEGANTLY
	return 1234567.89;
}

