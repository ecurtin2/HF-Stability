#include "HFS_fileIO.h"

namespace HFS {

    void print_params() {
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

}
