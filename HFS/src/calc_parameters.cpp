#include "calc_parameters.hpp"
#include "NDmap.hpp"

namespace HFS {
    void calcParameters() {
        HFS::kf = HFS::calcKf(HFS::rs, NDIM);
        HFS::kmax = 2.000001 * HFS::kf; // The offset from 2 helps remove coincidence
                                        // cases where the states fall exactly on kf
        HFS::bzone_length = 2.0 * HFS::kmax;
        HFS::fermi_energy = 0.5 * HFS::kf * HFS::kf;
        HFS::kgrid = arma::linspace(-HFS::kmax, HFS::kmax, HFS::Nk);
        HFS::deltaK = HFS::kgrid(1) - HFS::kgrid(0);
        HFS::calcStates(HFS::kgrid, HFS::Nk, NDIM,
                        HFS::Nocc, HFS::Nvir, HFS::N_elec, HFS::occ_states, HFS::vir_states);
        HFS::calcVolAndTwoEConst(HFS::N_elec, HFS::rs, HFS::vol, HFS::two_e_const);
        HFS::calcEnergies(HFS::occ_states, HFS::occ_energies);
        HFS::calcEnergies(HFS::vir_states, HFS::vir_energies);
        HFS::calcExcitations(HFS::kgrid, HFS::Nk, HFS::deltaK, HFS::Nocc, HFS::Nvir,
                             HFS::occ_states, HFS::vir_states,
                             HFS::excitations, HFS::exc_energies, HFS::Nexc);
        HFS::calcExcitationEnergies();
        HFS::calcLowestEnergyExcitationDegeneracy();
        HFS::calcVirNTo1Map();
        HFS::calcInverseExcitationMap();
    }

    double calcKf (double rs, unsigned ndim) {
        if (NDIM == 1) {
            kf = PI / (4.0 * rs);
        } else if (NDIM == 2) {
            kf = sqrt(2.0) / rs;
        } else if (NDIM == 3) {
            kf = std::pow((9.0 * PI / 4.0), (1.0/3.0)) * (1.0 / rs);
        }
        return kf;
    }

    void calcVolAndTwoEConst (unsigned N_elec, double rs, double& Vol, double& TwoEConst) {
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

    void calcStates(arma::vec& kgrid
                   ,unsigned Nk
                   ,unsigned ndim
                   ,arma::uword& Nocc
                   ,arma::uword& Nvir
                   ,arma::uword& N_elec
                   ,arma::umat& occStates
                   ,arma::umat& virStates
                   ) {
        arma::uword N = Nk - 1;  // Unique Brillioun Zone
        arma::uword Nrows = std::pow(N, NDIM);
        //states.set_size(Nrows, NDIM);
        arma::mat states(Nrows, NDIM);

        if (NDIM == 1) {
            for (arma::uword i = 0; i < N; ++i) {
                states(i) = kgrid(i);
            }
        } else if (NDIM == 2) {
            for (arma::uword i = 0; i < N; ++i) {
                for (arma::uword j = 0; j < N; ++j) {
                    states(N*i + j, 0) = kgrid(i);
                    states(N*i + j, 1) = kgrid(j);
                }
            }
        } else if (NDIM == 3) {
            for (arma::uword i = 0; i < N; ++i) {
                for (arma::uword j = 0; j < N; ++j) {
                    for (arma::uword k = 0; k < N; ++k) {
                        states(N*N*i + N*j + k, 0) = kgrid(i);
                        states(N*N*i + N*j + k, 1) = kgrid(j);
                        states(N*N*i + N*j + k, 2) = kgrid(k);
                    }
                }
            }
        }
        double row_norm;

        arma::uvec occ_indices(Nrows), vir_indices(Nrows);  // Allocate extra space to avoid append
        Nocc = 0;
        Nvir = 0;
        for (arma::uword i = 0; i < Nrows; ++i) {
            row_norm = arma::norm(states.row(i));
            if (HFS::isOccupied(row_norm)) {
                occ_indices(Nocc) = i;
                ++Nocc;
            } else {
                vir_indices(Nvir) = i;
                ++Nvir;
            }
        }

        occ_indices = occ_indices.head(HFS::Nocc); // Clip trailing elements
        vir_indices = vir_indices.head(HFS::Nvir);
        arma::mat occStateMomenta  = states.rows(occ_indices);
        arma::mat virStateMomenta  = states.rows(vir_indices);
        occStates = kToIndex(occStateMomenta).t();
        virStates = kToIndex(virStateMomenta).t();
        N_elec = 2 * HFS::Nocc;
    }

    void calcEnergies(arma::umat& inp_states, arma::vec& energy_vec) {
        arma::uword num_inp_states = inp_states.n_cols;
        energy_vec.set_size(num_inp_states);
        energy_vec.fill(0.0);
        for (arma::uword i = 0; i < num_inp_states; ++i) {
            for (unsigned j = 0; j < NDIM; ++j) {
                energy_vec(i) += HFS::kgrid(inp_states(j, i)) * HFS::kgrid(inp_states(j, i));
            }
            energy_vec[i] /= 2.0; //Is now filled with kinetic energy
            energy_vec[i] += HFS::exchange(inp_states, i);
        }
    }

    void calcExcitations(arma::vec& kgrid
                        ,unsigned Nk
                        ,double deltaK
                        ,arma::uword Nocc
                        ,arma::uword Nvir
                        ,arma::umat& occ_states
                        ,arma::umat& vir_states
                        ,arma::umat& excitations
                        ,arma::vec&  exc_energies
                        ,arma::uword& Nexc) {
        arma::vec kexc(NDIM);
        arma::uvec vir_idx(NDIM);
        arma::uvec exc_idx(NDIM);
        excitations.set_size(2, Nocc * Nvir);
        exc_energies.set_size(Nocc * Nvir);
        Nexc = 0;
        for (arma::uword i = 0; i < occ_states.n_cols; ++i) {
            // Excite only in +x direction
            for (arma::uword j = 1; j < Nk-1; ++j) {
                kexc = kgrid(occ_states.col(i));
                kexc(0) += deltaK * j;
                HFS::toFirstBrillouinZone(kexc);
                //exc_idx = HFS::kToIndex(kexc);
                HFS::kToIndex(kexc, exc_idx);
                // Find the vir state
                for (arma::uword k = 0; k < vir_states.n_cols; ++k) {
                    vir_idx = vir_states.col(k);
                    if (arma::all(exc_idx == vir_idx)) {
                        excitations(0, Nexc) = i;
                        excitations(1, Nexc) = k;
                        ++Nexc;
                    }
                }

            }
        }
        excitations  = excitations.head_cols(Nexc);
        exc_energies = exc_energies.head(Nexc);
    }

    void calcExcitationEnergies() {
        HFS::exc_energies.zeros(HFS::Nexc);
        for (arma::uword i = 0; i < HFS::Nexc; ++i) {
            HFS::exc_energies(i) = HFS::vir_energies(HFS::excitations(1, i))
                                 - HFS::occ_energies(HFS::excitations(0, i));
        }

        arma::uvec indices = arma::sort_index(HFS::exc_energies);
        HFS::exc_energies = HFS::exc_energies(indices);
        HFS::excitations = HFS::excitations.cols(indices);
    }

    /* NEED 3D VERSION THO*/

    #if NDIM == 2
        void calcVirNTo1Map() {
            HFS::vir_N_to_1_mat.set_size(HFS::Nk-1, HFS::Nk-1);
            HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
            for (arma::uword i=0; i < HFS::Nvir; ++i){
                HFS::vir_N_to_1_mat(HFS::vir_states(0, i), HFS::vir_states(1, i)) = i;
            }
        }
    #elif NDIM == 3
        void calcVirNTo1Map() {
        HFS::vir_N_to_1_mat.set_size(HFS::Nk-1, HFS::Nk-1, HFS::Nk-1);
        HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
        for (arma::uword i=0; i < HFS::Nvir; ++i){
            HFS::vir_N_to_1_mat(HFS::vir_states(0, i), HFS::vir_states(1, i), HFS::vir_states(2, i)) = i;
            }
        }
    #endif // NDIM

    void calcInverseExcitationMap() {
        HFS::inv_exc_mat.set_size(HFS::Nocc, HFS::Nvir);
        HFS::inv_exc_mat.fill(HFS::Nexc+1); // will make errors if accessing wrong one
        for (arma::uword i = 0; i < HFS::Nexc; ++i) {
            HFS::inv_exc_mat(HFS::excitations(0, i), HFS::excitations(1, i)) = i;
        }
    }

    void calcLowestEnergyExcitationDegeneracy() {
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
