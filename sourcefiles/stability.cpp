//Stability analysis, the state information is passed in from python
#include <iostream>
#include <iomanip>
#include <cmath>
#include <assert.h>
#include <time.h>
#include "stability.h"

void HFStability::HEG::calc_exc_energy() {
    exc_energies.zeros(Nexc);
    for (arma::uword i = 0; i < Nexc; ++i) { 
        exc_energies(i) = vir_energies(excitations(i, 1)) 
                        - occ_energies(excitations(i, 0));
    }
}

void HFStability::HEG::calc_energy_wrap(bool is_vir) {
    if (is_vir) {
        calc_energies(vir_states, vir_energies);
    }else{
        calc_energies(occ_states, occ_energies);
    }
}

void HFStability::HEG::calc_energies(arma::umat& inp_states, arma::vec& energy_vec) {
    arma::uword num_inp_states = inp_states.n_rows;
    energy_vec.set_size(num_inp_states);
    energy_vec.fill(0.0);
    for (arma::uword i = 0; i < num_inp_states; ++i) {
        for (int j = 0; j < ndim; ++j) {
            energy_vec(i) += kgrid(inp_states(i,j)) * kgrid(inp_states(i,j)); 
        }  
        energy_vec[i] /= 2.0; //Is now filled with kinetic energy
        energy_vec[i] += exchange(inp_states, i); 
    }
}

double HFStability::HEG::exchange(arma::umat& inp_states, arma::uword i) {

    double exch = 0.0;
    arma::vec ki(ndim), k2(ndim);
    

    for (int j = 0; j < ndim; ++j) {
        ki(j) = kgrid(inp_states(i, j));
    }

    for (arma::uword k = 0; k < Nocc; ++k) {
        for (int j = 0; j < ndim; ++j) {
            k2(j) = kgrid(occ_states(k, j));
        }

        exch += two_electron(ki, k2);
    } 

    exch *= -1.0;
    return exch;
}

double HFStability::HEG::two_electron(arma::vec k1, arma::vec k2) {
    double norm = 0.0;
    arma::vec k(ndim);
    k = k1 - k2;

    to_first_BZ(k);
    norm = arma::norm(k);
    if (norm < 10E-10) {
        return 0.0;
    }else{
        return two_e_const / std::pow(norm, ndim - 1);
    }
}

double HFStability::HEG::two_electron_check(arma::vec k1, arma::vec k2, arma::vec k3, arma::vec k4) {
    // Same as two_electron, except checks for momentum conservation
    // In the other, conservation is assumed
    double sum_sqrs = 0.0;
    arma::vec k(ndim);

    k = k1 + k2 - k3 - k4;
    to_first_BZ(k);

    // If not momentum conserving:
    if (arma::any(arma::abs(k) > 10E-10)) {
            return 0.0;
    }

    k =  k1 - k3;
    to_first_BZ(k);
    double norm = arma::norm(k);

    if (norm < 10E-10) {
        return 0.0;
    }else{
        return two_e_const / std::pow(norm, ndim - 1);
    }
}

void HFStability::HEG::get_vir_N_to_1_map() {
    for (arma::uword i = 0; i < Nvir; ++i) {
        std::vector<arma::uword> key(ndim);
        for (int j = 0; j < ndim; ++j) {
            key[j] = vir_states(i, j);
        }
        vir_N_to_1_map[key] = i;
    }
}

std::vector<arma::uword> HFStability::HEG::k_to_idx(arma::vec k) {
    arma::vec idx = arma::round((k + kmax) / deltaK);
    std::vector<arma::uword> indices = arma::conv_to<std::vector<arma::uword>>::from(idx);
    return indices;
}

arma::vec HFStability::HEG::occ_idx_to_k(arma::uword idx) {
    arma::vec k(ndim);
    for (int i = 0; i < ndim; ++i) {
        k[i] = kgrid(occ_states(idx, i));
    }
    return k;
}

arma::vec HFStability::HEG::vir_idx_to_k(arma::uword idx) {
    arma::vec k(ndim);
    for (int i = 0; i < ndim; ++i) {
        k[i] = kgrid(vir_states(idx, i));
    }
    return k;

}

void HFStability::HEG::get_inv_exc_map() { 
    for (arma::uword i = 0; i < Nexc; ++i) {
        std::vector<arma::uword> key {excitations(i,0), excitations(i,1)};
        inv_exc_map[key] = i;
    }

    // Testing 
    inv_exc_map_test.set_size(Nexc);
    for (arma::uword i = 0; i < Nexc; ++i) {
        std::vector<arma::uword> key {excitations(i,0), excitations(i,1)};
        inv_exc_map_test(i) = inv_exc_map[key];
    }
}

double HFStability::HEG::get_1B(arma::uword s, arma::uword t) {
    arma::uword i = excitations(s, 0);
    arma::uword a = excitations(s, 1);
    arma::uword j = excitations(t, 0);
    arma::uword b = excitations(t, 1);
    arma::vec ki(ndim), kj(ndim), ka(ndim), kb(ndim);
    for (int idx = 0; idx < ndim; ++idx) {
        ki[idx] = kgrid(occ_states(i, idx));
        kj[idx] = kgrid(occ_states(j, idx));
        ka[idx] = kgrid(vir_states(a, idx));
        kb[idx] = kgrid(vir_states(b, idx));
    }
    return 2.0 * two_electron_check(ka, kb, ki, kj) - two_electron_check(ka, kb, kj, ki);
}

double HFStability::HEG::get_3B(arma::uword s, arma::uword t) {
//    std::cout << "Start get_3A s =" << s << " t =" << t << std::endl; //DEBUG
    arma::uword i = excitations(s, 0);
    arma::uword a = excitations(s, 1);
    arma::uword j = excitations(t, 0);
    arma::uword b = excitations(t, 1);
    arma::vec ki(ndim), kj(ndim), ka(ndim), kb(ndim);
    for (int idx = 0; idx < ndim; ++idx) {
        ki[idx] = kgrid(occ_states(i, idx));
        kj[idx] = kgrid(occ_states(j, idx));
        ka[idx] = kgrid(vir_states(a, idx));
        kb[idx] = kgrid(vir_states(b, idx));
    }
//    std::cout << "Finish get_3A s =" << s << " t =" << t << std::endl; //DEBUG
    return -1.0 * two_electron_check(ka, kb, kj, ki);
}

double HFStability::HEG::get_1A(arma::uword s, arma::uword t) {
    arma::uword i = excitations(s, 0);
    arma::uword a = excitations(s, 1);
    arma::uword j = excitations(t, 0);
    arma::uword b = excitations(t, 1);
    arma::vec ki(ndim), kj(ndim), ka(ndim), kb(ndim);
    for (int idx = 0; idx < ndim; ++idx) {
        ki[idx] = kgrid(occ_states(i, idx));
        kj[idx] = kgrid(occ_states(j, idx));
        ka[idx] = kgrid(vir_states(a, idx));
        kb[idx] = kgrid(vir_states(b, idx));
    }
    double val = 0.0;
    if ((i == j) && (a == b)) {
        val = exc_energies(s);
    }
    val += 2.0 * two_electron_check(ka, kj, ki, kb) - two_electron_check(ka, kj, kb, ki);
    return val;
}

double HFStability::HEG::get_3A(arma::uword s, arma::uword t) {
//    std::cout << "Start get_3A s =" << s << " t =" << t << std::endl; //DEBUG
    arma::uword i = excitations(s, 0);
    arma::uword a = excitations(s, 1);
    arma::uword j = excitations(t, 0);
    arma::uword b = excitations(t, 1);
//    std::cout << "i =" << i << std::endl; //DEBUG 
//    std::cout << "j =" << j << std::endl; //DEBUG 
//    std::cout << "a =" << a << std::endl; //DEBUG 
//    std::cout << "b =" << b << std::endl; //DEBUG
    arma::vec ki(ndim), kj(ndim), ka(ndim), kb(ndim);
    for (int idx = 0; idx < ndim; ++idx) {
        ki[idx] = kgrid(occ_states(i, idx));
        kj[idx] = kgrid(occ_states(j, idx));
        ka[idx] = kgrid(vir_states(a, idx));
        kb[idx] = kgrid(vir_states(b, idx));
    }
    double val = 0.0;
    if ((i == j) && (a == b)) {
        val = exc_energies(s);
    }
    val += -1.0 * two_electron_check(ka, kj, kb, ki);
//    std::cout << "Finish get_3A s =" << s << " t =" << t << std::endl; //DEBUG
    return val;

}

double HFStability::HEG::get_3H(arma::uword i, arma::uword j) {
//    std::cout << "Start get_3H i =" << i << " j =" << j << std::endl; //DEBUG
    if (i < Nexc) {
        if (j < Nexc) {
            // First quadrant
//            std::cout << "1" << std::endl; //DEBUG
            return get_3A(i,j);
        }else{
            // Second quadrant
//            std::cout << "2" << std::endl; //DEBUG
            return get_3B(i, j-Nexc);
        }
    }else{
        if (j < Nexc) {
            // Third quadrant
//            std::cout << "3" << std::endl; //DEBUG
            return get_3B(i-Nexc,j);
        }else{
            // Second quadrant
//            std::cout << "4" << std::endl; //DEBUG
            return get_3A(i-Nexc, j-Nexc);
        }
    }
}

void HFStability::HEG::build_mattest() {
    mattest.set_size(2*Nexc, 2*Nexc);
    for (arma::uword i = 0; i < 2*Nexc; ++i) {
        for (arma::uword j = 0; j < 2*Nexc; ++j) {
            mattest(i,j) =  get_3H(i,j);
        }
    }
}

void HFStability::HEG::matvec_prod_arma() {
    out_vec1 = mattest * inp_test_vec;
}

void HFStability::HEG::matvec_prod_me() {
    out_vec2 = matvec_prod_3H(inp_test_vec);

}

void HFStability::HEG::to_first_BZ(arma::vec& k) {
    // Translate to first brillioun zone, defined on the
    // interval [-pi/a .. pi/a)  
    
    for (int i = 0; i < ndim; ++i) {
        if (k[i] < -kmax - 10E-10) {
            k[i] += bzone_length;
        }else if (k[i] > kmax - 10E-10) {
            k[i] -= bzone_length;
        }
    }
}

arma::vec HFStability::HEG::matvec_prod_3H(arma::vec v) {
    get_vir_N_to_1_map();
    get_inv_exc_map();

    arma::vec Mv(Nexc*2.0, arma::fill::zeros);  // matrix vector product
    /*  The matrix-vector multiplication | A  B | |v1|  =  | Mv1 |
                                         | B* A*| |v2|     | Mv2 |
        Factors into 4 matrix vector multiplications (x, y are vectors; A, B are matrices)
        Mv1 = Av1  + Bv2
        Mv2 = B*v1 + A*v2 
    */
    arma::vec v1 = v.head(Nexc);
    arma::vec v2 = v.tail(Nexc);
    arma::vec Mv1(Nexc, arma::fill::zeros), Mv2(Nexc, arma::fill::zeros);
    Mv1 = matvec_prod_3A(v1) + matvec_prod_3B(v2);
    Mv2 = matvec_prod_3B(v1) + matvec_prod_3A(v2);
    Mv = arma::join_cols(Mv1, Mv2);
    return Mv;
}

arma::uword HFStability::HEG::kb_j_to_t(arma::vec kb, arma::uword j) {
    std::vector<arma::uword> b_N_idx = k_to_idx(kb);
    arma::uword b = vir_N_to_1_map.at(b_N_idx);
    std::vector<arma::uword> key {j, b};
    arma::uword t = inv_exc_map.at(key);
    return t;
}

arma::vec HFStability::HEG::matvec_prod_3A(arma::vec v) {
    assert (v.n_elem == Nexc);
    arma::vec Mv(Nexc, arma::fill::zeros);
    for (arma::uword s = 0; s < Nexc; ++s) {
        arma::uword i = excitations(s, 0), a = excitations(s, 1);
        arma::vec ki(ndim), ka(ndim);
        ki = occ_idx_to_k(i);
        ka = vir_idx_to_k(a);
        for (arma::uword j = 0; j < Nocc; ++j) {
            arma::vec kj(ndim), kb(ndim);
            kj = occ_idx_to_k(j);
            kb = ka + kj - ki; // Momentum conservation for <aj|bi>
            to_first_BZ(kb);
            if (arma::norm(kb) > (kf + 10E-8)) {
                // only if momentum conserving state is virtual
                arma::uword t = kb_j_to_t(kb, j);
                if (s == t) {
                    Mv(s) += exc_energies(s) * v(t);
                } else { 
                    Mv(s) += -1.0 * two_electron(ka, kb) * v(t);
                }

            }
        }
    }
    return Mv;
}

arma::vec HFStability::HEG::matvec_prod_3B(arma::vec v) {
    assert (v.n_elem == Nexc);
    arma::vec Mv(Nexc, arma::fill::zeros);
    for (arma::uword s = 0; s < Nexc; ++s) {
        arma::uword i = excitations(s, 0), a = excitations(s, 1);
        arma::vec ki(ndim), ka(ndim);
        ki = occ_idx_to_k(i);
        ka = vir_idx_to_k(a);
        for (arma::uword j = 0; j < Nocc; ++j) {
            arma::vec kj(ndim), kb(ndim);
            kj = occ_idx_to_k(j); 
            kb = kj + ki - ka; // Momentum conservation for <ab|ji>
            to_first_BZ(kb);
            if (arma::norm(kb) > (kf + 10E-8)) {
                // only if momentum conserving state is virtual
                arma::uword t = kb_j_to_t(kb, j);
                Mv(s) += -1.0 * two_electron(ka, kj) * v(t);
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
