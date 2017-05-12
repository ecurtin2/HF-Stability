#include "parameters.hpp"

/** @file parameters.hpp
@author Evan Curtin
@version Revision 0.1
@brief Header including extern declarations for global parameters.
@details Definitions are in parameters.cpp
@date Wednesday, 04 Jan, 2017
*/

PhysicalParams::PhysicalParams(scalar inp_rs, std::string inp_mycase) {
        rs = inp_rs;
        mycase = inp_mycase;
        kf = calcKf(rs, NDIM);
        kmax = 2.000001 * kf; // The offset from 2 helps remove coincidence
                                        // cases where the states fall exactly on kf
        bzone_length = 2.0 * kmax;
        fermi_energy = 0.5 * kf * kf;
        kgrid = arma::linspace(-kmax, kmax, Nk);
        deltaK = kgrid(1) - kgrid(0);

        // The order of these cannot be changed.
        calcStates();
        calcVolAndTwoEConst();
        calcEnergiesForStates();
        calcEnergiesForStates();
        calcExcitations();
        calcExcitationEnergies();
        calcLowestEnergyExcitationDegeneracy();
        calcVirNTo1Map();
        calcInverseExcitationMap();
    }

scalar PhysicalParams::calcKf () {
    if (NDIM == 1) {
        kf = PI / (4.0 * rs);
    } else if (NDIM == 2) {
        kf = sqrt(2.0) / rs;
    } else if (NDIM == 3) {
        kf = std::pow((9.0 * PI / 4.0), (1.0/3.0)) * (1.0 / rs);
    }
    return kf;
}

void PhysicalParams::calcVolAndTwoEConst () {
    if (NDIM == 1) {
        vol = N_elec * rs;
        scalar a = 0.01;
        two_e_const = a * PI / vol;    // 'a' for the exponential integral, radius of cylinder.
    } else if (NDIM == 2) {
        vol = N_elec * PI * std::pow(rs, 2);
        two_e_const = 2.0 * PI / vol;
    } else if (NDIM == 3) {
        vol = N_elec * 4.0 / 3.0 * PI * std::pow(rs, 3);
        two_e_const = 4.0 * PI / vol;
    }
}

void PhysicalParams::calcStates() {
    uint N = Nk - 1;  // Unique Brillioun Zone
    uint Nrows = std::pow(N, NDIM);
    //states.set_size(Nrows, NDIM);
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

void PhysicalParams::calcEnergiesForStates() {
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

void PhysicalParams::calcExcitations() {
    arma::vec  kexc(NDIM);
    arma::uvec vir_uint(NDIM);
    arma::uvec exc_uint(NDIM);
    excitations.set_size(2, Nocc * Nvir);
    exc_energies.set_size(Nocc * Nvir);
    Nexc = 0;


    for (uint i = 0; i < occ_states.n_cols; ++i) {
        // Excite only in +x direction
        for (uint j = 1; j < Nk-1; ++j) {
            kexc = kgrid(occ_states.col(i));
            kexc(0) += deltaK * j;
            HFS::toFirstBrillouinZone(kexc);
            HFS::kToIndex(kexc, exc_uint);
            // Find the vir state
            for (uint k = 0; k < vir_states.n_cols; ++k) {
                vir_uint = vir_states.col(k);
                if (arma::all(exc_uint == vir_uint)) {
                    excitations(0, Nexc) = i;
                    excitations(1, Nexc) = k;
                    ++Nexc;
                }
            }

        }
    }
    excitations  = excitations.head_cols(Nexc); // clip size
    exc_energies = exc_energies.head(Nexc);  // clip size
}

void PhysicalParams::calcExcitationEnergies() {
    exc_energies.zeros(Nexc);
    for (uint i = 0; i < Nexc; ++i) {
        exc_energies(i) = vir_energies(excitations(1, i))
                             - occ_energies(excitations(0, i));
    }

    arma::uvec indices = arma::sort_index(exc_energies);
    exc_energies = exc_energies(indices);
    excitations = excitations.cols(indices);
}

# if NDIM == 1
    void PhysicalParams::calcVirNTo1Map() {
    HFS::vir_N_to_1_mat.set_size(HFS::Nk-1);
    HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
    for (uint i=0; i < HFS::Nvir; ++i){
        HFS::vir_N_to_1_mat(HFS::vir_states(i)) = i;
    }
}

# elif NDIM == 2
    void PhysicalParams::calcVirNTo1Map() {
        HFS::vir_N_to_1_mat.set_size(HFS::Nk-1, HFS::Nk-1);
        HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
        for (uint i=0; i < HFS::Nvir; ++i){
            HFS::vir_N_to_1_mat(HFS::vir_states(0, i), HFS::vir_states(1, i)) = i;
        }
    }
# elif NDIM == 3
    void calcVirNTo1Map() {
    HFS::vir_N_to_1_mat.set_size(HFS::Nk-1, HFS::Nk-1, HFS::Nk-1);
    HFS::vir_N_to_1_mat.fill(HFS::Nvir+1); // will make errors if accessing wrong one
    for (uint i=0; i < HFS::Nvir; ++i){
        HFS::vir_N_to_1_mat(HFS::vir_states(0, i), HFS::vir_states(1, i), HFS::vir_states(2, i)) = i;
        }
    }
#endif // NDIM

void PhysicalParams::calcInverseExcitationMap() {
    HFS::inv_exc_mat.set_size(HFS::Nocc, HFS::Nvir);
    HFS::inv_exc_mat.fill(HFS::Nexc+1); // will make errors if accessing wrong one
    for (uint i = 0; i < HFS::Nexc; ++i) {
        HFS::inv_exc_mat(HFS::excitations(0, i), HFS::excitations(1, i)) = i;
    }
}

void PhysicalParams::calcLowestEnergyExcitationDegeneracy() {
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


std::string PhysicalParams::to_json() {


#define JSONVAL(x) output << ",\n\"" << # x << "\" : " << x
#define JSONSTR(x) output << ",\n\"" << # x << "\" : \"" << x << "\""

    JSONVAL(Nk);
    JSONVAL(NDIM);
    JSONVAL(rs);
    JSONSTR(mycase);

    JSONVAL(deltaK);
    JSONVAL(kf);
    JSONVAL(kmax);
    JSONVAL(Nocc);
    JSONVAL(Nvir);
    JSONVAL(Nexc);
    JSONVAL(Nmat);
    JSONVAL(ground_state_degeneracy);

#undef JSONVAL
#undef JSONSTR


}

#endif // HFS_PARAMS_INCLUDED
