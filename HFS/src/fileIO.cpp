#include "fileIO.hpp"
#include <string>
#include <chrono>
#include <ctime>

#define JSONVAL(x) output << ",\n\"" << #x << "\" : " << x
#define JSONSTR(x) output << ",\n\"" << #x << "\" : \"" << x << "\""

namespace HFS {

    template <class T>
    void writeArmaMatToJSON(std::ofstream& output, const arma::Mat<T>& M, const std::string& varname) {
        output << ",\n\"" << varname << "\" : ";
        output << "[";

        arma::Mat<T> Mt = M.t();

        output << "[" << Mt(0, 0);
        for (arma::uword j = 1; j < Mt.n_rows; ++j){
                output << ", " << Mt(j, 0);
        }

        output << "]";
        for (arma::uword i = 1; i < Mt.n_cols; ++i) {

            // for each col of M
            output << ",\n[" << Mt(0, i);
            for (arma::uword j = 1; j < Mt.n_rows; ++j){
                    output << ", " << Mt(j, i);
            }
            output << "]";
        }

        output << "]";
    }

    template <class T>
    void writeArmaVecToJSON(std::ofstream& output, const arma::Col<T>& v, const std::string& varname) {
        output << ",\n\"" << varname << "\" : [" << v[0];
        for (arma::uword i = 1; i < v.n_elem; ++i) {
            output << ", " << v[i];
        }
        output << "]";
    }

    void writeJSON(std::string fname, bool detail) {
        std::ofstream output;

        output.open(fname.c_str());


        using namespace HFS;
        if (output.is_open() && output.good()) {


            output.precision(__DBL_DIG__);
            output.setf( std::ios::fixed, std:: ios::floatfield );

            std::chrono::time_point<std::chrono::system_clock> thetime;
            thetime = std::chrono::system_clock::now();
            std::time_t end_time = std::chrono::system_clock::to_time_t(thetime);


            std::string computation_finished = std::string(std::ctime(&end_time));

            std::string build_date = std::string(__DATE__) + "-" + std::string(__TIME__);

            std::ostringstream timeostringstream;
            timeostringstream << Total_Calculation_Time;
            std::string total_calculation_time = timeostringstream.str();
            Computation_Starttime.pop_back();
            computation_finished.pop_back();

            // Begin output
            output << "{\"File\" : \"" << fname << "\"";
            JSONSTR(total_calculation_time);
            JSONSTR(computation_finished);
            JSONSTR(build_date);
            JSONSTR(total_calculation_time);

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
            JSONVAL(ground_state_degeneracy);
            JSONVAL(dav_its);
            JSONVAL(num_guess_evecs);
            JSONVAL(Dav_blocksize);
            JSONVAL(Dav_Num_evals);
            JSONVAL(Mv_time);
            JSONVAL(cond_number);
            JSONVAL(Dav_nconv);

            if (fabs(full_diag_min) > 1e-6) {
                JSONVAL(full_diag_min);
                JSONVAL(full_diag_time);
            }

            JSONVAL(Dav_tol);
            JSONVAL(Dav_maxits);
            JSONVAL(Dav_maxsubsize);
            JSONVAL(Dav_final_val);
            JSONVAL(Dav_time);

            if (detail) {
                writeArmaVecToJSON(output, occ_energies, "occ_energies");
                writeArmaVecToJSON(output, vir_energies, "vir_energies");
                writeArmaVecToJSON(output, exc_energies, "exc_energies");
                writeArmaVecToJSON(output, kgrid, "kgrid");

                writeArmaMatToJSON(output, occ_states, "occ_states");
                writeArmaMatToJSON(output, vir_states, "vir_states");
                writeArmaMatToJSON(output, excitations, "excitations");
            }

            output << "}";

        }
        output.close();
    }
}
