//Stability analysis, the state information is passed in from python
#include <iostream>
#include <iomanip>
#include <cmath>
#include <assert.h>
#include <time.h>
#include "stability.h"
#define ARMA_NO_DEBUG

void HFStability::HEG::calc_exc_energy() {
    exc_energies.zeros(Nexc);
    for (arma::uword i = 0; i < Nexc; ++i) { 
        exc_energies(i) = vir_energies(excitations(i, 1)) 
                        - occ_energies(excitations(i, 0));
    }
}

void HFStability::HEG::calc_energy_wrap(bool is_vir) {
    if (ndim == 3) {
        if (is_vir) {
            calc_energies_3d(vir_states, vir_energies);
        }else{ 
            calc_energies_3d(occ_states, occ_energies);
        }

    } else if (ndim == 2) {
        if (is_vir) {
            calc_energies_2d(vir_states, vir_energies);
        }else{ 
            calc_energies_2d(occ_states, occ_energies);
        }
    }
}

void HFStability::HEG::calc_energies_2d(arma::umat& inp_states, arma::vec& energy_vec) {
    arma::uword num_inp_states = inp_states.n_rows;
    energy_vec.set_size(num_inp_states);
    energy_vec.fill(0.0);
    for (arma::uword i = 0; i < num_inp_states; ++i) {
        for (unsigned int j = 0; j < ndim; ++j) {
            energy_vec(i) += kgrid(inp_states(i,j)) * kgrid(inp_states(i,j)); 
        }  
        energy_vec[i] /= 2.0; //Is now filled with kinetic energy
        energy_vec[i] += exchange_2d(inp_states, i); 
    }
}

double HFStability::HEG::exchange_2d(arma::umat& inp_states, arma::uword i) {
    double exch = 0.0;
    double ki[2] = {kgrid(inp_states(i, 0)), kgrid(inp_states(i,1))};

    for (arma::uword j = 0; j < Nocc; ++j) {
        double kj[2] = {kgrid(occ_states(j, 0)), kgrid(occ_states(j,1))};
        exch += two_electron_2d(ki, kj);
    } 
    exch *= -1.0;
    return exch;
}

double HFStability::HEG::two_electron_2d(double k1[], double k2[]) {
    double sum_sqrs = 0.0;
    double k[2];

    for (unsigned int i = 0; i < 2; ++i) { 
        k[i] = k1[i] - k2[i];
        //Shift into first brillouin zone
        if (k[i] < -kmax) {
            k[i] += bzone_length;
        }else if (k[i] > kmax) {
            k[i] -= bzone_length;
        }
        sum_sqrs += k[i] * k[i];
    }

    if (sum_sqrs < 10E-10) {
        return 0.0;
    }else{
        return two_e_const / sqrt(sum_sqrs);
    }
}

void HFStability::HEG::calc_energies_3d(arma::umat& inp_states, arma::vec& energy_vec) {
    arma::uword num_inp_states = inp_states.n_rows;
    energy_vec.set_size(num_inp_states);
    energy_vec.fill(0.0);
    for (arma::uword i = 0; i < num_inp_states; ++i) {
        for (unsigned int j = 0; j < ndim; ++j) {
            energy_vec(i) += kgrid(inp_states(i,j)) * kgrid(inp_states(i,j)); 
        }  
        energy_vec[i] /= 2.0; //Is now filled with kinetic energy
        energy_vec[i] += exchange_3d(inp_states, i); 
    }
}

double HFStability::HEG::exchange_3d(arma::umat& inp_states, arma::uword i) {
    double exch = 0.0;
    double ki[3] = {kgrid(inp_states(i, 0)), kgrid(inp_states(i,1)), kgrid(inp_states(i,2))};

    for (arma::uword j = 0; j < Nocc; ++j) {
        double kj[3] = {kgrid(occ_states(j, 0)), kgrid(occ_states(j,1)), kgrid(occ_states(j,2))};
        exch += two_electron_3d(ki, kj);
    } 
    exch *= -1.0;
    return exch;
}

double HFStability::HEG::two_electron_3d(double k1[], double k2[]) {
    double sum_sqrs = 0.0;
    double k[3];

    for (unsigned int i = 0; i < 3; ++i) { 
        k[i] = k1[i] - k2[i];
        //Shift into first brillouin zone
        if (k[i] < -kmax) {
            k[i] += bzone_length;
        }else if (k[i] > kmax) {
            k[i] -= bzone_length;
        }
        sum_sqrs += k[i] * k[i];
    }
    if (sum_sqrs < 10E-10) {
        return 0.0;
    }else{
        return two_e_const / sum_sqrs;
    }
}

void HFStability::HEG::get_vir_states_inv_2d() {
//arma::umat vir_states_inv(3)
    for (arma::uword i = 0; i < Nvir; ++i) {
        int x = 0;
    }
}

arma::uword HFStability::HEG::k_to_idx(double k[]) {
    
    for (arma::uword i = 0; i < Nexc; ++i) { 
        int k = 1;
    }
    return 3.0;
}

void HFStability::HEG::get_inv_exc_map_2d() { 
    for (arma::uword i = 0; i < Nexc; ++i) {
        auto key = std::make_tuple(excitations(i,0), excitations(i,1));
        inv_exc_map_2d[key] = i;
    }
    inv_exc_map_2d_test.set_size(Nexc);
    for (arma::uword i = 0; i < Nexc; ++i) {
        arma::uword j = excitations(i, 0);
        arma::uword k = excitations(i, 1);
        auto key = std::make_tuple(j, k);
        inv_exc_map_2d_test(i) = inv_exc_map_2d[key];
    }
}


arma::vec& HFStability::HEG::mat_vec_prod_2d(arma::vec v) {
    // so the convention for orbital indices can be used
    // orbital indices will be of the form i_A, where i is the 
    // orbital index and the  _A means this corresponds to the A matrix

    arma::vec Mv(Nexc*2.0, arma::fill::zeros);  // matrix vector product
    
    // This outer loop fills Mv[0..Nexc]
    for (arma::uword s = 0; s < Nexc; ++s) {
        // s, i and a are the same for A and B (Seeger/Pople notation)
        arma::uword i = excitations(s, 0);
        arma::uword a = excitations(s, 1);
        double ki[2] = {kgrid(occ_states(i, 0)), kgrid(occ_states(i, 1))};
        double ka[2] = {kgrid(occ_states(a, 0)), kgrid(occ_states(a, 1))};
        for (arma::uword j = 0; j < Nocc; ++j) {
            // j is the same for A and B
            double kj[2] = {kgrid(occ_states(j, 0)), kgrid(occ_states(j, 1))};
            // find the momentum conserving virtual orbital
            double kb_A[2] = { (ki[0] + ka[0] - kj[0]), (ki[1] + ka[1] - kj[1]) };
            double kb_B[2] = { (ki[0] + kj[0] - ka[0]), (ki[1] + kj[1] - ka[1]) };
            // virtual orbital index, k_to_idx needs to first find indices of kgrid, then do an inverse
            // map of vir_states
            arma::uword b_A = 1;
            arma::uword b_B = 1;
//            arma::uword b_A = k_to_idx(kb_A);
//            arma::uword b_B = k_to_idx(kb_B);
            // excitations index, needed to know matrix location; A[s,t]
            arma::uword t_A = inverse_exc_map(j, b_A);
            arma::uword t_B = inverse_exc_map(j, b_A);

            Mv(s) += two_electron_2d(kj, ka) * v(t_B + Nexc); // B is offset by Nexc because of the layout of H
            if (s == t_A) { // if diagonal element of A
                Mv(s) += exc_energies(s) * v(t_A);
            } else { // off-diagonal of A
                Mv(s) += two_electron_2d(kb_A, ka) * v(t_A);
            }
            
            //This is the second half of the vector
            Mv(s + Nexc) += two_electron_2d(kj, ka) * v(t_B);
            if (s == t_A) { // if diagonal element of A
                Mv(s + Nexc) += exc_energies(s) * v(t_A + Nexc);
            } else { // off-diagonal of A
                Mv(s + Nexc) += two_electron_2d(kb_A, ka) * v(t_A + Nexc);
            }


        }
    }
    return Mv;
}

double HFStability::HEG::davidson_algorithm(
    uint64_t N,
    uint64_t max_its, 
    uint64_t max_sub_size,
    uint64_t num_of_roots,
    arma::uword block_size,
    arma::mat(guess_evecs),
    double tolerance, 
    double (HFStability::HEG::*matrix)(uint64_t, uint64_t)) {
    
    uint64_t sub_size = guess_evecs.n_cols;
    uint64_t old_sub_size = 0;    
    arma::uword num_new_vecs = sub_size;
    arma::vec old_evals;
    arma::mat old_evecs = guess_evecs;
    arma::mat mat_vec_prod(N, 0, arma::fill::zeros);
    arma::mat ritz_vecs = guess_evecs;

    
    arma::mat init_guess(N, num_of_roots);
    for (uint64_t i = 0; i < num_of_roots; ++i) {
        init_guess.col(i) = guess_evecs.col(i);
    }

        

    //Iterate the block Davidson algorithm.
    for (uint64_t i=0 ; i < max_its ; ++i){
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
        
        //Get the res and append to subspace if needed.
        //Sub_size changes within this loop
        old_sub_size = sub_size;
        arma::vec norms(block_size, arma::fill::zeros);
        for (uint64_t j=0; j < block_size; ++j){
            arma::vec res(N, arma::fill::zeros);
            double rayq = 0.0;
            double x_k = 0.0;
            //rayleigh quotient is x.T * M * x
            for (uint64_t k = 0; k < N; ++k) {
                x_k = ritz_vecs(k,j);    
                for (uint64_t l = 0; l < N; ++l) {
                    rayq += x_k * (this->*matrix)(k,l) * ritz_vecs(l,j);
                }
            }

            for (uint64_t k=0; k < N; ++k) {
                sum = 0.0;
                for (uint64_t m=0; m < N; ++m) {    
                    if (m == k){
                        sum += ((this->*matrix)(k,k) - rayq) * ritz_vecs(m,j);
                    }else{
                        sum += (this->*matrix)(k,m) * ritz_vecs(m,j);
                    }
                res(k) = sum;
                }
            }
        
            //Append only the res with norm > tolerance.
            double norm = arma::norm(res);
            norms(j) = norm;
            if (norm > tolerance) {
                arma::vec corr(N, arma::fill::zeros);

                //If diagonal element = eval, get singularity
                bool apply_corr = true;
                for (uint64_t k = 0; k < old_sub_size; ++k) {
                    if ( fabs( (this->*matrix)(k,k) - sub_evals(j) ) <= 10E-10) {
                        apply_corr = false;
                    }
                }

                //This is the DPR corr
                for (uint64_t k=0 ; k < N ; ++k) {
                    corr(k) = ( -1.0/ ( (this->*matrix)(k,k) - rayq ) ) * res(k);
                }

                //Only add if no problems encountered
                if (apply_corr) {
                    corr /= arma::norm(corr);
                    guess_evecs = arma::join_rows(guess_evecs, corr);
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
            std::cout << "Davidson: Subspace converged in " << i + 1 << 
            " iterations!"  << std::endl;
            return sub_evals(0);

        }else if ( arma::all(norms.rows(0,num_of_roots-1)  < tolerance)) {
            std::cout << "Davidson: All requested norms converged in " 
            << i + 1 << " iterations!"  << std::endl;
            return sub_evals(0);
        }            

        //Make sure the new guess space is orthonormal
        arma::qr_econ(Q, R, guess_evecs);
        //Enforce sums of elements of eigenvector matrix are positive
        //with no loss of generality. 
        for (uint64_t i = 0; i < sub_size; ++i) {
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

