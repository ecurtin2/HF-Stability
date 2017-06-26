#include "calc_parameters.hpp"
#include "NDmap.hpp"
#include <set>
#include <algorithm>

namespace HFS {


    void calcParameters() {
        HFS::kf = HFS::calcKf(HFS::rs, NDIM);
        HFS::kmax = 2.000001 * HFS::kf; // The offset from 2 helps remove coincidence
                                        // cases where the states fall exactly on kf
        HFS::bzone_length = 2.0 * HFS::kmax;
        HFS::fermi_energy = 0.5 * HFS::kf * HFS::kf;
        HFS::kgrid = arma::linspace(-HFS::kmax, HFS::kmax, HFS::Nk + 1);
        HFS::kgrid.shed_row(HFS::Nk);
        HFS::deltaK = HFS::kgrid(1) - HFS::kgrid(0);
        HFS::calcStates(HFS::kgrid, HFS::Nk, NDIM,
                        HFS::Nocc, HFS::Nvir, HFS::N_elec, HFS::occ_states, HFS::vir_states);
        HFS::calcVolAndTwoEConst(HFS::N_elec, HFS::rs, HFS::vol, HFS::two_e_const);
        HFS::calcEnergies(HFS::occ_states, HFS::occ_energies);
        HFS::calcEnergies(HFS::vir_states, HFS::vir_energies);

        // Sort occ and vir by energies
        /*auto idx = arma::sort_index(HFS::occ_energies);
        HFS::occ_energies = HFS::occ_energies(idx);
        HFS::occ_states = HFS::occ_states.rows(idx);

        auto vir_idx = arma::sort_index(HFS::vir_energies);
        HFS::vir_energies = HFS::vir_energies(idx);
        HFS::vir_states = HFS::vir_states.rows(idx);
*/

        HFS::calcExcitations(HFS::kgrid, HFS::Nk, HFS::deltaK, HFS::Nocc, HFS::Nvir,
                             HFS::occ_states, HFS::vir_states,
                             HFS::excitations, HFS::exc_energies, HFS::Nexc);
        HFS::Nvir = HFS::vir_states.n_cols;
        HFS::calcExcitationEnergies();
        HFS::calcLowestEnergyExcitationDegeneracy();
        HFS::calcVirNTo1Map();
        HFS::calcInverseExcitationMap();
    }

    scalar calcKf (scalar rs, uint ndim) {
        if (NDIM == 1) {
            kf = PI / (4.0 * rs);
        } else if (NDIM == 2) {
            kf = sqrt(2.0) / rs;
        } else if (NDIM == 3) {
            kf = std::pow((9.0 * PI / 4.0), (1.0/3.0)) * (1.0 / rs);
        }
        return kf;
    }

    void calcVolAndTwoEConst (uint N_elec, scalar rs, scalar& Vol, scalar& TwoEConst) {
        if (NDIM == 1) {
            vol = 2.0 * N_elec * rs;
            if (use_delta_1D == true) {
                two_e_const = twoE_parameter_1dCase;
            } else {
                two_e_const = twoE_parameter_1dCase * PI / vol;    // 'a' for the exponential integral, radius of cylinder.
            }
        } else if (NDIM == 2) {
            vol = N_elec * PI * std::pow(rs, 2);
            two_e_const = 2.0 * PI / vol;
        } else if (NDIM == 3) {
            vol = N_elec * 4.0 / 3.0 * PI * std::pow(rs, 3);
            two_e_const = 2.0 * PI / vol;
        }
    }

    void calcStates(arma::vec& kgrid
                   ,uint Nk
                   ,uint ndim
                   ,uint& Nocc
                   ,uint& Nvir
                   ,uint& N_elec
                   ,arma::umat& occStates
                   ,arma::umat& virStates
                   ) {
        uint N = Nk;
        uint Nrows = std::pow(N, NDIM);
        arma::mat states(Nrows, NDIM);

        if (NDIM == 1) {
            for (uint i = 0; i < N; ++i) {
                states(i) = kgrid(i);
            }
        } else if (NDIM == 2) {
            for (uint i = 0; i < N; ++i) {
                for (uint j = 0; j < N; ++j) {
                    states(N*i + j, 0) = kgrid(i);
                    states(N*i + j, 1) = kgrid(j);
                }
            }
        } else if (NDIM == 3) {
            for (uint i = 0; i < N; ++i) {
                for (uint j = 0; j < N; ++j) {
                    for (uint k = 0; k < N; ++k) {
                        states(N*N*i + N*j + k, 0) = kgrid(i);
                        states(N*N*i + N*j + k, 1) = kgrid(j);
                        states(N*N*i + N*j + k, 2) = kgrid(k);
                    }
                }
            }
        }
        scalar row_norm;

        arma::uvec occ_indices(Nrows), vir_indices(Nrows);  // Allocate extra space to avoid append
        Nocc = 0;
        Nvir = 0;
        for (uint i = 0; i < Nrows; ++i) {
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
        uint num_inp_states = inp_states.n_cols;
        energy_vec.set_size(num_inp_states);
        energy_vec.fill(0.0);
        for (uint i = 0; i < num_inp_states; ++i) {
            for (uint j = 0; j < NDIM; ++j) {
                energy_vec(i) += HFS::kgrid(inp_states(j, i)) * HFS::kgrid(inp_states(j, i));
            }
            energy_vec[i] /= 2.0; //Is now filled with kinetic energy
            energy_vec[i] += HFS::exchange(inp_states, i);
        }
    }

    void calcExcitations(arma::vec& kgrid
                        ,uint Nk
                        ,scalar deltaK
                        ,uint Nocc
                        ,uint Nvir
                        ,arma::umat& occ_states
                        ,arma::umat& vir_states
                        ,arma::umat& excitations
                        ,arma::vec&  exc_energies
                        ,uint& Nexc) {

        excitations.set_size(2, Nocc * Nvir);
        exc_energies.set_size(Nocc * Nvir);
        Nexc = 0;


        std::set<uint> unique_virs;
        for (uint i = 0; i < occ_states.n_cols; ++i) {
            for (uint a = 0; a < vir_states.n_cols; ++a) {
                bool momentum_conserving = true;
                // Check over all non-x dimensions, and only add if they are the same.
                for (uint k = 1; k < NDIM; ++k) {
                    if (occ_states(k, i) != vir_states(k, a)) {
                        momentum_conserving = false;
                    }
                }
                if (momentum_conserving) {
                    unique_virs.insert(a);
                    excitations(0, Nexc) = i;
                    excitations(1, Nexc) = a;
                    ++Nexc;
                }
            }
        }


        excitations  = excitations.head_cols(Nexc); // clip size
        exc_energies = exc_energies.head(Nexc);  // clip size

        // I hate this but this is the way that this worked -
        // removing all the unused virtuals and changing excitations to refer to the new indices.
        for (uint i = 0; i < excitations.n_cols; ++i) {
            // remap the excitation indices to the new virtual indices
            excitations(1, i) = std::distance(unique_virs.begin(), unique_virs.find(excitations(1, i)));
        }

        arma::uvec unique_virs_vec(unique_virs.size());
        uint idx = 0;
        for (auto it : unique_virs) {
            unique_virs_vec(idx) = it;
            ++idx;
        }
        vir_states = vir_states.cols(unique_virs_vec);
        HFS::vir_energies = HFS::vir_energies(unique_virs_vec);
    }

    void calcExcitationEnergies() {
        HFS::exc_energies.zeros(HFS::Nexc);
        for (uint i = 0; i < HFS::Nexc; ++i) {
            HFS::exc_energies(i) = HFS::vir_energies(HFS::excitations(1, i))
                                 - HFS::occ_energies(HFS::excitations(0, i));
        }

        arma::uvec indices = arma::sort_index(HFS::exc_energies);
        HFS::exc_energies = HFS::exc_energies(indices);
        HFS::excitations = HFS::excitations.cols(indices);
    }

    # if NDIM == 1
        void calcVirNTo1Map() {
        HFS::vir_N_to_1_mat.set_size(HFS::Nk);
        HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
        for (uint i=0; i < HFS::Nvir; ++i){
            HFS::vir_N_to_1_mat(HFS::vir_states(i)) = i;
        }
    }

    # elif NDIM == 2
        void calcVirNTo1Map() {
            HFS::vir_N_to_1_mat.set_size(HFS::Nk, HFS::Nk);
            HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
            for (uint i=0; i < HFS::Nvir; ++i){
                HFS::vir_N_to_1_mat(HFS::vir_states(0, i), HFS::vir_states(1, i)) = i;
            }
        }
    # elif NDIM == 3
        void calcVirNTo1Map() {
        HFS::vir_N_to_1_mat.set_size(HFS::Nk, HFS::Nk, HFS::Nk);
        HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
        for (uint i=0; i < HFS::Nvir; ++i){
            HFS::vir_N_to_1_mat(HFS::vir_states(0, i), HFS::vir_states(1, i), HFS::vir_states(2, i)) = i;
            }
        }
    #endif // NDIM

    void calcInverseExcitationMap() {
        HFS::inv_exc_mat.set_size(HFS::Nocc, HFS::Nvir);
        HFS::inv_exc_mat.fill(HFS::Nexc+1); // will make errors if accessing wrong one
        for (uint i = 0; i < HFS::Nexc; ++i) {
            HFS::inv_exc_mat(HFS::excitations(0, i), HFS::excitations(1, i)) = i;
        }
    }

    void calcLowestEnergyExcitationDegeneracy() {
        scalar lowest_energy = HFS::exc_energies(0);
        HFS::ground_state_degeneracy = 1;
        for (uint i = 1; i < HFS::Nexc; ++i) {
            if (HFS::exc_energies(i) < (lowest_energy + SMALLNUMBER) ) {
                HFS::ground_state_degeneracy += 1;
            } else {
                break;
            }
        }
    }

} // namespace HFS
