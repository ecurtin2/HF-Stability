#include "fileIO.hpp"
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>

#define PRINTVAL(x) std::cout << #x << " = " << x << std::endl;
#define PRINT(x) std::cout << x << std::endl;
#define WIDTH 80
#define HASHTAGLINE std::cout << std::string(WIDTH, '#') << std::endl;
#define PRINTLINE std::cout << std::string(WIDTH, '-') << std::endl;
#define NEWLINE std::cout << std::endl;
#define CENTERED(x) centerString(x, WIDTH)
#define ENDSECTION(x) NEWLINE PRINT(CENTERED("End of Section: " + std::string(x)))
#define SECTION(x) NEWLINE PRINTLINE NEWLINE PRINT(CENTERED(x)) NEWLINE PRINTLINE NEWLINE

namespace HFS {



    void writeOutput(bool detail) {
        using namespace HFS;
        std::cout.precision(__DBL_DIG__);
        std::cout.setf( std::ios::fixed, std:: ios::floatfield );

        HASHTAGLINE NEWLINE
        PRINT(CENTERED("Output File for HFS, the Hartree-Fock Stability Program"))
        PRINT(CENTERED("Author: Evan Curtin"))
        PRINT(CENTERED("Institution: University of Illinois @ Urbana-Champaign"))
        PRINT(CENTERED("email: ecurtin2@illinois.edu"))
        PRINT(CENTERED( "Built on " + std::string(__DATE__) + ", at " + std::string(__TIME__) ))
        NEWLINE HASHTAGLINE


        // Get the End time
        std::chrono::time_point<std::chrono::system_clock> thetime;
        thetime = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(thetime);
        std::string Computation_Finished = std::string(std::ctime(&end_time));

        // Time stuff
        NEWLINE
        PRINT(CENTERED("Computation Started : " + Computation_Starttime))
        PRINT(CENTERED("Computation Finished: " + Computation_Finished))
        std::ostringstream timeostringstream;
        timeostringstream << Total_Calculation_Time;
        std::string timestring = timeostringstream.str();
        PRINT(CENTERED(("Total Elapsed Time = " + timestring + " seconds.")))

        SECTION("Input")
        PRINTVAL(Nk)
        PRINTVAL(NDIM)
        PRINTVAL(rs)
        PRINTVAL(mycase)
        ENDSECTION("Input")

        SECTION("Output")
        PRINTVAL(deltaK)
        PRINTVAL(kf)
        PRINTVAL(kmax)
        PRINTVAL(Nocc)
        PRINTVAL(Nvir)
        PRINTVAL(Nexc)
        PRINTVAL(ground_state_degeneracy)
        PRINTVAL(dav_its)
        PRINTVAL(num_guess_evecs)
        PRINTVAL(Dav_blocksize)
        PRINTVAL(Dav_Num_evals)
        PRINTVAL(Mv_time)
        PRINTVAL(cond_number)
        PRINTVAL(Dav_nconv)

        std::cout.precision(2);
        std::cout << "Davidson Tolerance = " << std::scientific << Dav_tol << std::fixed << std::endl;
        std::cout.precision(__DBL_DIG__);
        PRINTVAL(Dav_maxits)
        PRINTVAL(Dav_maxsubsize)
        PRINTVAL(Dav_final_val)
        PRINTVAL(Dav_time)

        assert (HFS::everything_works());

        if (Nk < 31){
            PRINTVAL(full_diag_min)
            PRINTVAL(full_diag_time)
        }

        ENDSECTION("Output")

        if (detail) {
            SECTION("Vectors")
            std::cout << "Occ Energies: " << occ_energies.n_rows << std::endl; occ_energies.raw_print();
            std::cout << "Vir Energies: " << vir_energies.n_rows << std::endl; vir_energies.raw_print();
            std::cout << "Excitation Energies: " << exc_energies.n_rows << std::endl; exc_energies.raw_print();
            std::cout << "Kgrid: " << kgrid.n_rows << std::endl; kgrid.raw_print();

            //std::cout << "DavVals: " << dav_vals.n_rows << std::endl; dav_vals.raw_print();
            ENDSECTION("Vectors")

            SECTION("Matrices")
            std::cout << "Occupied States: " << occ_states.n_rows << " x " << occ_states.n_cols << std::endl;
            occ_states.print();
            std::cout << "Virtual States: " << vir_states.n_rows << " x " << vir_states.n_cols << std::endl;
            vir_states.print();
            std::cout << "Excitations: " << excitations.n_rows << " x " << excitations.n_cols << std::endl;
            excitations.print();
            std::setw(0);
            ENDSECTION("Matrices")
        }

        NEWLINE HASHTAGLINE NEWLINE
        PRINT(CENTERED("End of Output File."))
        NEWLINE HASHTAGLINE

    }

    std::string centerString(std::string s, int width) {
        int pad = (width-s.size()) / 2;
        std::string pads = std::string( pad, ' ' );
        std::string centered =  pads + s + pads;
        return centered;
    }

}
