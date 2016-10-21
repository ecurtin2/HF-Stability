#include "HFS_debug.hpp"

namespace HFS{

    arma::mat full_matrix;
    double full_diag_min;
    double Mv_time;
    double full_diag_time;
    void time_mv();

    bool davidson_agrees_fulldiag() {
        arma::wall_clock timer;
        timer.tic();

        HFS::build_matrix();
        arma::vec eigvals;
        arma::mat eigvecs;
        arma::eig_sym(eigvals, eigvecs, HFS::full_matrix);
        HFS::full_diag_min = eigvals.min();
        HFS::full_diag_time = timer.toc();

        arma::vec last_row = dav_vals.tail_rows(0);
        double diff = fabs(arma::min(eigvals) - arma::min(last_row));
        bool agrees = (diff < 10E-5);
        return agrees;
    }

    bool mv_is_working() {
        arma::vec v(2*HFS::Nexc, arma::fill::randu);
        arma::vec Mv = HFS::matvec_prod_3H(v);
        HFS::build_matrix();
        arma::vec v_arma = HFS::full_matrix * v;
        arma::vec diff = arma::abs(Mv - v_arma);
        bool is_working = arma::all(diff < SMALLNUMBER);
        return is_working;
    }

    void time_mv() {
        arma::wall_clock timer;
        timer.tic();
        arma::vec v(2*HFS::Nexc, arma::fill::randu);
        arma::vec Mv = HFS::matvec_prod_3H(v);
        HFS::Mv_time = timer.toc();
    }

    void build_matrix() {
        HFS::full_matrix.set_size(2*HFS::Nexc, 2*HFS::Nexc);
        for (arma::uword i = 0; i < 2*HFS::Nexc; ++i) {
            for (arma::uword j = 0; j < 2*HFS::Nexc; ++j) {
                HFS::full_matrix(i,j) =  HFS::calc_3H(i,j);
            }
        }
    }

    bool everything_works() {
        assert(mv_is_working() && "There is a problem in the matrix-vector product.");
        std::cout << "Mv is working" << std::endl;
        assert(davidson_agrees_fulldiag() && "Davidson's Algorithm Didn't get the lowest eigenvalue.");
        std::cout << "Davidson is working" << std::endl;
        return true;
    }

}
