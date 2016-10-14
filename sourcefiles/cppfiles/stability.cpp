//Stability analysis, the state information is passed in from python
#include <iostream>
#include <cmath>
#include <assert.h>
#include <time.h>
#include "HFSnamespace.h"

//Parameter Calculation

void HFS::calc_params() {
    HFS::calc_kf();
    HFS::kmax = 2.0 * HFS::kf;
    HFS::bzone_length = 2.0 * HFS::kmax;
    HFS::fermi_energy = 0.5 * HFS::kf * HFS::kf;
    HFS::kgrid = arma::linspace(-HFS::kmax, HFS::kmax, HFS::Nk);
    HFS::deltaK = HFS::kgrid(1) - HFS::kgrid(0);
    HFS::calc_occ_states();
    HFS::calc_vol_and_two_e_const();
    HFS::calc_occ_energies();
    HFS::calc_vir_energies();
    HFS::calc_excitations();
    HFS::calc_vir_N_to_1_mat();
    HFS::calc_inv_exc_mat();
}

void HFS::calc_kf () {
    if (HFS::ndim == 1) {
        HFS::kf = PI / (4.0 * HFS::rs);
    } else if (HFS::ndim == 2) {
        HFS::kf = sqrt(2.0) / HFS::rs;
    } else if (HFS::ndim == 3) {
        HFS::kf = std::pow((9.0 * PI / 4.0), (1.0/3.0)) * (1.0 / HFS::rs);
    }
}

void HFS::calc_vol_and_two_e_const () {
    if (HFS::ndim == 1) {
        HFS::vol = HFS::N_elec * 2.0 * HFS::rs;
    } else if (HFS::ndim == 2) {
        HFS::vol = HFS::N_elec * PI * std::pow(HFS::rs, 2);
        HFS::two_e_const = 2.0 * PI / HFS::vol;
    } else if (HFS::ndim == 3) {
        HFS::vol = HFS::N_elec * 4.0 / 3.0 * PI * std::pow(HFS::rs, 3);
        HFS::two_e_const = 4.0 * PI / HFS::vol;
    }
}

void HFS::calc_occ_states() {
    arma::uword N = Nk - 1;  // Unique Brillioun Zone
    arma::uword Nrows = std::pow(N, HFS::ndim);
    //states.set_size(Nrows, HFS::ndim);
    arma::mat states(Nrows, HFS::ndim);

    if (HFS::ndim == 1) {
        for (arma::uword i = 0; i < N; ++i) {
            states(i) = HFS::kgrid(i);
        }
    } else if (HFS::ndim == 2) {
        for (arma::uword i = 0; i < N; ++i) {
            for (arma::uword j = 0; j < N; ++j) {
                states(N*i + j, 0) = HFS::kgrid(i);
                states(N*i + j, 1) = HFS::kgrid(j);
            }
        }
    } else if (HFS::ndim == 3) {
        for (arma::uword i = 0; i < N; ++i) {
            for (arma::uword j = 0; j < N; ++j) {
                for (arma::uword k = 0; k < N; ++k) {
                    states(N*N*i + N*j + k, 0) = HFS::kgrid(i);
                    states(N*N*i + N*j + k, 1) = HFS::kgrid(j);
                    states(N*N*i + N*j + k, 2) = HFS::kgrid(k);
                }
            }
        }
    }
    double row_norm;

    arma::uvec occ_indices(Nrows), vir_indices(Nrows);  // Allocate extra space to avoid append
    HFS::Nocc = 0;
    HFS::Nvir = 0;
    for (arma::uword i = 0; i < Nrows; ++i) {
        row_norm = arma::norm(states.row(i));
        if (HFS::is_vir(row_norm)) {
            occ_indices(HFS::Nocc) = i;
            ++HFS::Nocc;
        } else {
            vir_indices(HFS::Nvir) = i;
            ++HFS::Nvir;
        }
    }

    occ_indices = occ_indices.head(HFS::Nocc); // Clip trailing elements
    vir_indices = vir_indices.head(HFS::Nvir);
    arma::mat occ_states  = states.rows(occ_indices);
    arma::mat vir_states  = states.rows(vir_indices);
    HFS::occ_states = k_to_index(occ_states);
    HFS::vir_states = k_to_index(vir_states);
    HFS::N_elec = 2 * HFS::Nocc;
}

void HFS::calc_occ_energies() {
    HFS::calc_energies(HFS::occ_states, HFS::occ_energies);
}

void HFS::calc_vir_energies() {
    HFS::calc_energies(HFS::vir_states, HFS::vir_energies);
}

void HFS::calc_energies(arma::umat& inp_states, arma::vec& energy_vec) {
    arma::uword num_inp_states = inp_states.n_rows;
    energy_vec.set_size(num_inp_states);
    energy_vec.fill(0.0);
    for (arma::uword i = 0; i < num_inp_states; ++i) {
        for (int j = 0; j < HFS::ndim; ++j) {
            energy_vec(i) += HFS::kgrid(inp_states(i,j)) * HFS::kgrid(inp_states(i,j));
        }
        energy_vec[i] /= 2.0; //Is now filled with kinetic energy
        energy_vec[i] += HFS::exchange(inp_states, i);
    }
}

void HFS::calc_excitations() {
    arma::vec kexc(HFS::ndim);
    arma::uvec vir_idx(HFS::ndim);
    arma::uvec exc_idx(HFS::ndim);
    HFS::excitations.set_size(HFS::Nocc * HFS::Nvir, 2);
    HFS::exc_energies.set_size(HFS::Nocc * HFS::Nvir);
    HFS::Nexc = 0;
    for (arma::uword i = 0; i < HFS::occ_states.n_rows; ++i) {
        // Excite only in +x direction
        for (arma::uword j = 1; j < HFS::Nk-1; ++j) {
            kexc = HFS::kgrid(HFS::occ_states.row(i));
            kexc(0) += HFS::deltaK * j;
            HFS::to_first_BZ(kexc);
            exc_idx = HFS::k_to_index(kexc);
            // Find the vir state
            for (arma::uword k = 0; k < HFS::vir_states.n_rows; ++k) {
                vir_idx = HFS::vir_states.row(k).t();
                if (arma::all(exc_idx == vir_idx)) {
                    HFS::excitations(HFS::Nexc, 0) = i;
                    HFS::excitations(HFS::Nexc, 1) = k; //THIS SHOULDNT BE K IT SHOULD BNE Soemth else
                    HFS::exc_energies(HFS::Nexc) = HFS::vir_energies(k) - HFS::occ_energies(i);
                    ++HFS::Nexc;
                }
            }

        }
    }
    HFS::excitations  = HFS::excitations.head_rows(HFS::Nexc);
    HFS::exc_energies = HFS::exc_energies.head(HFS::Nexc);
}

void HFS::calc_exc_energy() {
    HFS::exc_energies.zeros(HFS::Nexc);
    for (arma::uword i = 0; i < HFS::Nexc; ++i) {
        HFS::exc_energies(i) = HFS::vir_energies(HFS::excitations(i, 1))
                             - HFS::occ_energies(HFS::excitations(i, 0));
    }
}

/* NEED 3D VERSION THO*/
void HFS::calc_vir_N_to_1_mat() {
    HFS::vir_N_to_1_mat.set_size(HFS::Nk-1, HFS::Nk-1);
    HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
    for (arma::uword i=0; i < HFS::Nvir; ++i){
        HFS::vir_N_to_1_mat(HFS::vir_states(i, 0), HFS::vir_states(i, 1)) = i;
    }
}

void HFS::calc_inv_exc_mat() {
    HFS::inv_exc_mat.set_size(HFS::Nocc, HFS::Nvir);
    HFS::inv_exc_mat.fill(HFS::Nexc+1); // will make errors if accessing wrong one
    for (arma::uword i = 0; i < HFS::Nexc; ++i) {
        HFS::inv_exc_mat(HFS::excitations(i,0), HFS::excitations(i,1)) = i;
    }
}

//Common functions needed multiple places

double HFS::exchange(arma::umat& inp_states, arma::uword i) {

    double exch = 0.0;
    arma::vec ki(ndim), k2(ndim);
    for (int j = 0; j < HFS::ndim; ++j) {
        ki(j) = HFS::kgrid(inp_states(i, j));
    }
    for (arma::uword k = 0; k < HFS::Nocc; ++k) {
        for (int j = 0; j < HFS::ndim; ++j) {
            k2(j) = HFS::kgrid(HFS::occ_states(k, j));
        }
        exch += HFS::two_electron(ki, k2);
    }
    exch *= -1.0;
    return exch;
}

double HFS::two_electron(arma::vec& k1, arma::vec& k2) {
    double norm = 0.0;
    arma::vec k(HFS::ndim);
    k = k1 - k2;

    HFS::to_first_BZ(k);
    norm = arma::norm(k);
    if (norm < 10E-10) {
        return 0.0;
    }else{
        return HFS::two_e_const / std::pow(norm, HFS::ndim - 1);
    }
}

double HFS::two_electron_check(arma::vec& k1, arma::vec& k2, arma::vec& k3, arma::vec& k4) {
    // Same as two_electron, except checks for momentum conservation
    // In the other, conservation is assumed
    arma::vec k(HFS::ndim);

    k = k1 + k2 - k3 - k4;
    HFS::to_first_BZ(k);

    // If not momentum conserving:
    if (arma::any(arma::abs(k) > SMALLNUMBER)) {
            return 0.0;
    }

    k =  k1 - k3;
    HFS::to_first_BZ(k);
    double norm = arma::norm(k);

    if (norm < SMALLNUMBER) {
        return 0.0;
    }else{
        return HFS::two_e_const / std::pow(norm, HFS::ndim - 1);
        //return HFS::two_e_const / norm;   //  < 1% speedup in davidson, keep general
    }
}

void HFS::to_first_BZ(arma::vec& k) {
    // Translate to first brillioun zone, defined on the
    // interval [-pi/a .. pi/a)

    for (int i = 0; i < HFS::ndim; ++i) {
        if (k[i] < -HFS::kmax - SMALLNUMBER) {
            k[i] += HFS::bzone_length;
        }else if (k[i] > HFS::kmax - SMALLNUMBER) {
            k[i] -= HFS::bzone_length;
        }
    }
}

bool HFS::is_vir(double k) {
        return (k <= HFS::kf + SMALLNUMBER);
}

arma::uvec HFS::k_to_index(arma::vec& k) {
    arma::vec idx = arma::round((k + HFS::kmax) / HFS::deltaK);
    arma::uvec indices = arma::conv_to<arma::uvec>::from(idx);
    return indices;
}

arma::umat HFS::k_to_index(arma::mat& k) {
    arma::mat idx = arma::round((k + HFS::kmax) / HFS::deltaK);
    arma::umat indices = arma::conv_to<arma::umat>::from(idx);
    return indices;
}

arma::vec HFS::occ_idx_to_k(arma::uword idx) {
    arma::vec k(HFS::ndim);
    for (int i = 0; i < HFS::ndim; ++i) {
        k[i] = HFS::kgrid(HFS::occ_states(idx, i));
    }
    return k;
}

arma::vec HFS::vir_idx_to_k(arma::uword idx) {
    arma::vec k(HFS::ndim);
    for (int i=0; i < HFS::ndim; ++i) {
        k[i] = HFS::kgrid(vir_states(idx, i));
    }
    return k;

}

// Matrix and Matrix-Vector Products

double HFS::calc_1B(arma::uword s, arma::uword t) {
    arma::uword i =  HFS::excitations(s, 0);
    arma::uword a =  HFS::excitations(s, 1);
    arma::uword j =  HFS::excitations(t, 0);
    arma::uword b =  HFS::excitations(t, 1);
    arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
    for (int idx = 0; idx < HFS::ndim; ++idx) {
        ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
        kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
        ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
        kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
    }
    return 2.0 * two_electron_check(ka, kb, ki, kj) - two_electron_check(ka, kb, kj, ki);
}

double HFS::calc_3B(arma::uword s, arma::uword t) {
//    std::cout << "Start get_3A s =" << s << " t =" << t << std::endl; //DEBUG
    arma::uword i = HFS::excitations(s, 0);
    arma::uword a = HFS::excitations(s, 1);
    arma::uword j = HFS::excitations(t, 0);
    arma::uword b = HFS::excitations(t, 1);
    arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
    for (int idx = 0; idx < ndim; ++idx) {
        ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
        kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
        ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
        kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
    }
//    std::cout << "Finish get_3A s =" << s << " t =" << t << std::endl; //DEBUG
    return -1.0 * HFS::two_electron_check(ka, kb, kj, ki);
}

double HFS::calc_1A(arma::uword s, arma::uword t) {
    arma::uword i = HFS::excitations(s, 0);
    arma::uword a = HFS::excitations(s, 1);
    arma::uword j = HFS::excitations(t, 0);
    arma::uword b = HFS::excitations(t, 1);
    arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
    for (int idx = 0; idx < HFS::ndim; ++idx) {
        ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
        kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
        ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
        kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
    }
    double val = 0.0;
    if ((i == j) && (a == b)) {
        val = HFS::exc_energies(s);
    }
    val += 2.0 * HFS::two_electron_check(ka, kj, ki, kb) - HFS::two_electron_check(ka, kj, kb, ki);
    return val;
}

double HFS::calc_3A(arma::uword s, arma::uword t) {
    arma::uword i = HFS::excitations(s, 0);
    arma::uword a = HFS::excitations(s, 1);
    arma::uword j = HFS::excitations(t, 0);
    arma::uword b = HFS::excitations(t, 1);

    arma::vec ki(HFS::ndim), kj(HFS::ndim), ka(HFS::ndim), kb(HFS::ndim);
    for (int idx = 0; idx < HFS::ndim; ++idx) {
        ki[idx] = HFS::kgrid(HFS::occ_states(i, idx));
        kj[idx] = HFS::kgrid(HFS::occ_states(j, idx));
        ka[idx] = HFS::kgrid(HFS::vir_states(a, idx));
        kb[idx] = HFS::kgrid(HFS::vir_states(b, idx));
    }
    double val = 0.0;
    if ((i == j) && (a == b)) {
        val = HFS::exc_energies(s);
    }
    val += -1.0 * HFS::two_electron_check(ka, kj, kb, ki);

    return val;

}

double HFS::calc_3H(arma::uword i, arma::uword j) {
    if (i < HFS::Nexc) {
        if (j < HFS::Nexc) {
            // First quadrant
            return HFS::calc_3A(i, j);
        }else{
            // Second quadrant
            return HFS::calc_3B(i, j - HFS::Nexc);
        }
    }else{
        if (j < HFS::Nexc) {
            // Third quadrant
            return HFS::calc_3B(i - HFS::Nexc, j);
        }else{
            // Fourth quadrant
            return HFS::calc_3A(i-HFS::Nexc, j-HFS::Nexc);
        }
    }
}

arma::vec HFS::matvec_prod_3A(arma::vec& v) {
    arma::vec Mv(HFS::Nexc, arma::fill::zeros);
    for (arma::uword s = 0; s < HFS::Nexc; ++s) {
        arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
        arma::vec ki(HFS::ndim), ka(HFS::ndim);
        ki = HFS::occ_idx_to_k(i);
        ka = HFS::vir_idx_to_k(a);
        for (arma::uword j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(HFS::ndim), kb(HFS::ndim);
            kj = HFS::occ_idx_to_k(j);
            kb = ka + kj - ki; // Momentum conservation for <aj|bi>
            HFS::to_first_BZ(kb);
            if (arma::norm(kb) > (HFS::kf + 10E-8)) {
                // only if momentum conserving state is virtual
                arma::uword t = HFS::kb_j_to_t(kb, j);
                if (s == t) {
                    Mv(s) += HFS::exc_energies(s) * v(t);
                } else {
                    Mv(s) += -1.0 * HFS::two_electron(ka, kb) * v(t);
                }

            }
        }
    }
    return Mv;
}

arma::vec HFS::matvec_prod_3B(arma::vec& v) {
    arma::vec Mv(HFS::Nexc, arma::fill::zeros);
    for (arma::uword s = 0; s < HFS::Nexc; ++s) {
        arma::uword i = HFS::excitations(s, 0), a = HFS::excitations(s, 1);
        arma::vec ki(HFS::ndim), ka(HFS::ndim);
        ki = HFS::occ_idx_to_k(i);
        ka = HFS::vir_idx_to_k(a);
        for (arma::uword j = 0; j < HFS::Nocc; ++j) {
            arma::vec kj(HFS::ndim), kb(HFS::ndim);
            kj = HFS::occ_idx_to_k(j);
            kb = kj + ki - ka; // Momentum conservation for <ab|ji>
            HFS::to_first_BZ(kb);
            if (arma::norm(kb) > (HFS::kf + SMALLNUMBER)) {
                // only if momentum conserving state is virtual
                arma::uword t = HFS::kb_j_to_t(kb, j);
                Mv(s) += -1.0 * HFS::two_electron(ka, kj) * v(t);
            }
        }
    }
    return Mv;
}

arma::vec HFS::matvec_prod_3H(arma::vec& v) {
    /*  The matrix-vector multiplication | A  B | |v1|  =  | Mv1 |
                                         | B* A*| |v2|     | Mv2 |
        Factors into 4 matrix vector multiplications (x, y are vectors; A, B are matrices)
        Mv1 = A*v1 + B*v2
        Mv2 = B*v1 + A*v2
    */
    arma::vec v1 = v.head(HFS::Nexc);
    arma::vec v2 = v.tail(HFS::Nexc);

    arma::vec Mv1 = HFS::matvec_prod_3A(v1) + HFS::matvec_prod_3B(v2);
    arma::vec Mv2 = HFS::matvec_prod_3B(v1) + HFS::matvec_prod_3A(v2);
    arma::vec Mv = arma::join_cols(Mv1, Mv2);
    return Mv;
}

arma::uword HFS::kb_j_to_t(arma::vec& kb, arma::uword j) {
    //std::vector<arma::uword> b_N_idx(HFS::ndim);
    //b_N_idx = HFS::k_to_idx(kb);
    arma::uvec b_N_idx =  k_to_index(kb);
    //arma::uword b = HFS::vir_N_to_1_map(b_N_idx);
    //std::cout << "b = " << b << std::endl;
    arma::uword b = HFS::vir_N_to_1_mat(b_N_idx(0), b_N_idx(1));
    //std::cout << "b2 = " << b << std::endl;
    //std::vector<arma::uword> key {j, b};
    //arma::uword t = HFS::inv_exc_map.at(key);
    arma::uword t = HFS::inv_exc_mat(j, b);
    return t;
}

//Davidson Algorithm
void HFS::davidson_wrapper(arma::uword N
                          ,arma::mat   guess_evecs
                          ,arma::uword block_size
                          ,int         which
                          ,arma::uword num_of_roots
                          ,arma::uword max_its
                          ,arma::uword max_sub_size
                          ,double      tolerance
                          )
{
    davidson_algorithm(N, max_its, max_sub_size, num_of_roots, block_size, guess_evecs, tolerance,
                       &HFS::calc_3H,
                       &HFS::matvec_prod_3H);
}

void HFS::build_guess_evecs (int N, int which) {
    if (which == 0) {
        HFS::guess_evecs = arma::eye<arma::mat>(2*HFS::Nexc, N);
    } else if (which == 1) {
        HFS::guess_evecs = arma::eye<arma::mat>(2*HFS::Nexc, N);
    }
}

void HFS::davidson_algorithm(arma::uword N
                                         ,arma::uword max_its
                                         ,arma::uword max_sub_size
                                         ,arma::uword num_of_roots
                                         ,arma::uword block_size
                                         ,arma::mat&   guess_evecs
                                         ,double      tolerance
                                         ,double      (*matrix)(arma::uword, arma::uword)
                                         ,arma::vec   (*matvec_product)(arma::vec& v)
                                         )
{

    arma::uword sub_size = guess_evecs.n_cols;
    arma::uword old_sub_size = 0;
    arma::uword num_new_vecs = sub_size;
    arma::vec old_evals;
    arma::mat old_evecs = guess_evecs;
    arma::mat Mvmat(N, 0, arma::fill::zeros);
    arma::mat ritz_vecs = guess_evecs;
    arma::mat init_guess(N, num_of_roots);

    clock_t t, t1, t2, t3;

    //Iterate the block Davidson algorithm.
    for (arma::uword i = 0 ; i < max_its ; ++i){
        t1 = clock();
        arma::mat sub_mat(sub_size, sub_size, arma::fill::zeros);
        num_new_vecs = sub_size - old_sub_size;
        t = clock();
        for (arma::uword j = 0; j < num_new_vecs; ++j)
        {
            arma::vec guess_vec = guess_evecs.col(old_sub_size + j);
            arma::vec Matvec = matvec_product(guess_vec);
            Mvmat = arma::join_rows(Mvmat, Matvec);
        }
        t2 = clock() - t;
        std::cout << "Mv took " << ((float)t2) / CLOCKS_PER_SEC << " seconds" << std::endl;
        sub_mat = guess_evecs.t() * Mvmat;
        //Diagonalize subspace matrix.
        arma::vec sub_evals;
        arma::mat sub_evecs;
        arma::eig_sym(sub_evals, sub_evecs, sub_mat);

        //sort eigenvals, vecs by value
        arma::uvec indices = arma::sort_index(sub_evals);
        sub_evecs = sub_evecs.cols(indices);
        sub_evals = sub_evals.elem(indices);

        //old_evecs = sub_evecs;
        ritz_vecs = guess_evecs * sub_evecs;
        /*Current implementation is the Diagonally Preconditioned Residue (DPR) method.
          As far as I know right now, this is the original method propossed by Davidson
          Apparently, sometimes people simply call this the Davidson method. */

        //Get the res and append to subspace if needed.
        //Sub_size changes within this loop
        double sum;
        old_sub_size = sub_size;
        arma::vec norms(num_of_roots, arma::fill::zeros);
        for (arma::uword j = 0; j < block_size; ++j){
            arma::vec res(N, arma::fill::zeros);
            double rayq = 0.0;
            double x_k = 0.0;
            //rayleigh quotient is x.T * M * x
            for (arma::uword k = 0; k < N; ++k) {
                x_k = ritz_vecs(k,j);
                for (arma::uword l = 0; l < N; ++l) {
                    rayq += x_k * matrix(k,l) * ritz_vecs(l,j);
                }
            }

            for (arma::uword k = 0; k < N; ++k) {
                sum = 0.0;
                for (arma::uword m = 0; m < N; ++m) {
                    if (m == k){
                        sum += (matrix(k,k) - rayq) * ritz_vecs(m,j);
                    }else{
                        sum += matrix(k,m) * ritz_vecs(m,j);
                    }
                res(k) = sum;
                }
            }

            //Append only the res with norm > tolerance.
            double norm = arma::norm(res);
            if (j < num_of_roots) {
                norms(j) = norm;
            }
            if (norm > tolerance) {
                arma::vec corr(N, arma::fill::zeros);

                //If diagonal element = eval, get singularity
                bool apply_corr = true;
                for (arma::uword k = 0; k < old_sub_size; ++k) {
                    if ( fabs( matrix(k,k) - sub_evals(j) ) <= 10E-10) {
                        apply_corr = false;
                    }
                }

                //This is the DPR corr
                for (arma::uword k=0; k < N; ++k) {
                    double val = matrix(k,k) -  rayq;
                    corr(k) = -1.0 / val * res(k);
                }

                //Only add if no problems encountered
                if (apply_corr) {
                    corr /= arma::norm(corr);
                    guess_evecs = arma::join_rows(guess_evecs, corr);
                    sub_size += 1;
                }
            }
        }

        //Make sure the new guess space is orthonormal
        arma::mat Q, R;
        arma::qr_econ(Q, R, guess_evecs);
        //Enforce sums of elements of eigenvector matrix are positive
        //with no loss of generality.
        for (arma::uword i = 0; i < Q.n_cols; ++i) {
            if (arma::sum(Q.col(i)) < 0.0) {
                Q.col(i) = -1.0 * Q.col(i);
            }
        }
        guess_evecs = Q;

        HFS::dav_vecs =  guess_evecs;
        HFS::dav_vals = sub_evals;
        HFS::dav_its  = i;

        t3 = clock() - t1;
        std::cout << "Iteration took " << ((float)t3) / CLOCKS_PER_SEC << " seconds" << std::endl;
        std::cout << "min eigval = " << dav_vals.min() << std::endl;

        if (old_sub_size == sub_size) {
            HFS::dav_message = "Subspace Converged";
            break;
        }
        else if ((sub_size + block_size) > max_sub_size) {
            HFS::dav_message = "Subspace Size Too Large";
            break;
        } else if (arma::all(norms < tolerance)) {
            HFS::dav_message = "All requested norms below tolerance";
            break;
        }
    }
}

// Testing/Debugging Functions

bool HFS::davidson_agrees_fulldiag() {
    HFS::build_matrix();
    arma::vec eigvals;
    arma::mat eigvecs;
    clock_t t, t2;
    t = clock();
    arma::eig_sym(eigvals, eigvecs, HFS::full_matrix);
    t2 = clock() - t;
    std::cout << "Full took " << ((float)t2) / CLOCKS_PER_SEC << " seconds" << std::endl;
    double diff = fabs(arma::min(eigvals) - arma::min(HFS::dav_vals));
    bool agrees = (diff < 10E-5);
    return agrees;
}

bool HFS::mv_is_working() {
    arma::vec v(2*HFS::Nexc, arma::fill::randu);
    arma::vec Mv = HFS::matvec_prod_3H(v);
    HFS::build_matrix();
    arma::vec v_arma = HFS::full_matrix * v;
    arma::vec diff = arma::abs(Mv - v_arma);
    bool is_working = arma::all(diff < SMALLNUMBER);
    return is_working;
}

void HFS::build_matrix() {
    HFS::full_matrix.set_size(2*HFS::Nexc, 2*HFS::Nexc);
    for (arma::uword i = 0; i < 2*HFS::Nexc; ++i) {
        for (arma::uword j = 0; j < 2*HFS::Nexc; ++j) {
            HFS::full_matrix(i,j) =  HFS::calc_3H(i,j);
        }
    }
}

bool HFS::everything_works() {
    assert(mv_is_working() && "There is a problem in the matrix-vector product.");
    std::cout << "Mv is working" << std::endl;
    assert(davidson_agrees_fulldiag() && "Davidson's Algorithm Didn't get the lowest eigenvalue.");
    std::cout << "Davidson is working" << std::endl;
    return true;
}

// Output Control

void HFS::print_params() {
    std::cout << "DeltaK = " << HFS::deltaK << std::endl;
    std::cout << "Nk = " << HFS::Nk << std::endl;
    std::cout << "ndim = " << HFS::ndim << std::endl;
    std::cout << "rs = " << HFS::rs << std::endl;
    std::cout << "kf = " << HFS::kf << std::endl;
    std::cout << "kmax = " << HFS::kmax << std::endl;
    std::cout << "Nocc = " << HFS::Nocc << std::endl;
    std::cout << "Nvir = " << HFS::Nvir << std::endl;
    std::cout << "Nexc = " << HFS::Nexc << std::endl;
    std::cout << "DavIts=" << HFS::dav_its << std::endl;
    std::cout << "Smallest Eval = " << HFS::dav_vals.min() << std::endl;
    std::cout << HFS::dav_message << std::endl;
    if (Nk < 10) {
    HFS::kgrid.print("Kgrid");
    HFS::occ_states.print("Occupied States");
    HFS::vir_states.print("Virtual States");
    HFS::occ_energies.print("Occ Energy");
    HFS::vir_energies.print("Vir Energy");
    HFS::excitations.print("Excitations");
    }
}


