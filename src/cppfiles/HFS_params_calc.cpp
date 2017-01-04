#include "HFS_params_calc.hpp"

namespace HFS {
    void calc_params() {
        HFS::calc_kf();
        HFS::kmax = 2.000001 * HFS::kf; // The offset from 2 helps remove coincidence
                                        // cases where the states fall exactly on kf
        HFS::bzone_length = 2.0 * HFS::kmax;
        HFS::fermi_energy = 0.5 * HFS::kf * HFS::kf;
        HFS::kgrid = arma::linspace(-HFS::kmax, HFS::kmax, HFS::Nk);
        HFS::deltaK = HFS::kgrid(1) - HFS::kgrid(0);
        HFS::calc_occ_states();
        HFS::calc_vol_and_two_e_const();
        HFS::calc_occ_energies();
        HFS::calc_vir_energies();
        HFS::calc_excitations();
        HFS::calc_exc_energy();
        HFS::calc_ground_state_degeneracy();
        HFS::calc_vir_N_to_1_mat();
        HFS::calc_inv_exc_mat();
    }

    void calc_kf () {
        if (NDIM == 1) {
            HFS::kf = PI / (4.0 * HFS::rs);
        } else if (NDIM == 2) {
            HFS::kf = sqrt(2.0) / HFS::rs;
        } else if (NDIM == 3) {
            HFS::kf = std::pow((9.0 * PI / 4.0), (1.0/3.0)) * (1.0 / HFS::rs);
        }
    }

    void calc_vol_and_two_e_const () {
        if (NDIM == 1) {
            HFS::vol = HFS::N_elec * 2.0 * HFS::rs;
        } else if (NDIM == 2) {
            HFS::vol = HFS::N_elec * PI * std::pow(HFS::rs, 2);
            HFS::two_e_const = 2.0 * PI / HFS::vol;
        } else if (NDIM == 3) {
            HFS::vol = HFS::N_elec * 4.0 / 3.0 * PI * std::pow(HFS::rs, 3);
            HFS::two_e_const = 4.0 * PI / HFS::vol;
        }
    }

    void calc_occ_states() {
        arma::uword N = Nk - 1;  // Unique Brillioun Zone
        arma::uword Nrows = std::pow(N, NDIM);
        //states.set_size(Nrows, NDIM);
        arma::mat states(Nrows, NDIM);

        if (NDIM == 1) {
            for (arma::uword i = 0; i < N; ++i) {
                states(i) = HFS::kgrid(i);
            }
        } else if (NDIM == 2) {
            for (arma::uword i = 0; i < N; ++i) {
                for (arma::uword j = 0; j < N; ++j) {
                    states(N*i + j, 0) = HFS::kgrid(i);
                    states(N*i + j, 1) = HFS::kgrid(j);
                }
            }
        } else if (NDIM == 3) {
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
            if (HFS::is_occ(row_norm)) {
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

    void calc_occ_energies() {
        HFS::calc_energies(HFS::occ_states, HFS::occ_energies);
    }

    void calc_vir_energies() {
        HFS::calc_energies(HFS::vir_states, HFS::vir_energies);
    }

    void calc_energies(arma::umat& inp_states, arma::vec& energy_vec) {
        arma::uword num_inp_states = inp_states.n_rows;
        energy_vec.set_size(num_inp_states);
        energy_vec.fill(0.0);
        for (arma::uword i = 0; i < num_inp_states; ++i) {
            for (unsigned j = 0; j < NDIM; ++j) {
                energy_vec(i) += HFS::kgrid(inp_states(i,j)) * HFS::kgrid(inp_states(i,j));
            }
            energy_vec[i] /= 2.0; //Is now filled with kinetic energy
            energy_vec[i] += HFS::exchange(inp_states, i);
        }
    }

    void calc_excitations() {
        arma::vec kexc(NDIM);
        arma::uvec vir_idx(NDIM);
        arma::uvec exc_idx(NDIM);
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
                        HFS::excitations(HFS::Nexc, 1) = k;
                        ++HFS::Nexc;
                    }
                }

            }
        }
        HFS::excitations  = HFS::excitations.head_rows(HFS::Nexc);
        HFS::exc_energies = HFS::exc_energies.head(HFS::Nexc);
    }

    void calc_exc_energy() {
        HFS::exc_energies.zeros(HFS::Nexc);
        for (arma::uword i = 0; i < HFS::Nexc; ++i) {
            HFS::exc_energies(i) = HFS::vir_energies(HFS::excitations(i, 1))
                                 - HFS::occ_energies(HFS::excitations(i, 0));
        }

        arma::uvec indices = arma::sort_index(HFS::exc_energies);
        HFS::exc_energies = HFS::exc_energies(indices);
        HFS::excitations = HFS::excitations.rows(indices);
    }

    /* NEED 3D VERSION THO*/

    #if NDIM == 2
        void calc_vir_N_to_1_mat() {
            HFS::vir_N_to_1_mat.set_size(HFS::Nk-1, HFS::Nk-1);
            HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
            for (arma::uword i=0; i < HFS::Nvir; ++i){
                HFS::vir_N_to_1_mat(HFS::vir_states(i, 0), HFS::vir_states(i, 1)) = i;
            }
        }
    #elif NDIM == 3
        void calc_vir_N_to_1_mat() {
        HFS::vir_N_to_1_mat.set_size(HFS::Nk-1, HFS::Nk-1, HFS::Nk-1);
        HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
        for (arma::uword i=0; i < HFS::Nvir; ++i){
            HFS::vir_N_to_1_mat(HFS::vir_states(i, 0), HFS::vir_states(i, 1), HFS::vir_states(i, 2)) = i;
        }
        }
    #endif // NDIM

    void calc_inv_exc_mat() {
        HFS::inv_exc_mat.set_size(HFS::Nocc, HFS::Nvir);
        HFS::inv_exc_mat.fill(HFS::Nexc+1); // will make errors if accessing wrong one
        for (arma::uword i = 0; i < HFS::Nexc; ++i) {
            HFS::inv_exc_mat(HFS::excitations(i,0), HFS::excitations(i,1)) = i;
        }
    }

    void calc_ground_state_degeneracy() {
        double lowest_energy = HFS::exc_energies(0);
        HFS::ground_state_degeneracy = 1;
        for (arma::uword i = 1; i < HFS::Nexc; ++i) {
            if (HFS::exc_energies(i) < (lowest_energy + SMALLNUMBER) ) {
                HFS::ground_state_degeneracy += 1;
            } else {
                break;
            }
        }
    }

} // namespace HFS
