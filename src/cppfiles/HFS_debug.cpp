#include "HFS_debug.hpp"

namespace HFS{

    double full_diag_min;
    double Mv_time;
    double full_diag_time;

    bool davidson_agrees_fulldiag() {
        arma::wall_clock timer;
        timer.tic();

        arma::mat matrix = HFS::build_matrix(HFS::Matrix_func, HFS::Nmat);
        arma::vec eigvals;
        arma::mat eigvecs;
        arma::eig_sym(eigvals, eigvecs, matrix);
        HFS::full_diag_min = eigvals.min();
        HFS::full_diag_time = timer.toc();
        return true;
    }

    bool mv_is_working() {
        arma::vec v(HFS::Nmat, arma::fill::ones);
        arma::vec Mv(HFS::Nmat);
        HFS::MatVecProduct_func(v, Mv);
        arma::mat matrix = HFS::build_matrix(HFS::Matrix_func, HFS::Nmat);
        arma::vec v_arma = matrix * v;
        arma::vec diff = arma::abs(Mv - v_arma);

        #ifndef NDEBUG
            v_arma.print("full");
            Mv.print("mv");
            diff.print("diff");
            arma::vec v2(2*HFS::Nexc, arma::fill::ones);
            arma::mat Aprime = HFS::build_matrix(HFS::calc_Aprime, 2*HFS::Nexc);
            arma::mat Bprime = HFS::build_matrix(HFS::calc_Bprime, 2*HFS::Nexc);
            arma::vec mv2(2*HFS::Nexc);
            arma::vec mvfull2(2*HFS::Nexc);
            arma::vec diff2(2*HFS::Nexc);
            mv2 = HFS::matvec_prod_Aprime(v2);
            mvfull2 = Aprime * v2;
            diff2 = arma::abs(mv2 - mvfull2);
            diff2.print("diff for Aprime");
            mv2 = HFS::matvec_prod_Bprime(v2);
            mvfull2 = Bprime * v2;
            diff2 = arma::abs(mv2 - mvfull2);
            diff2.print("diff for Bprime");
        #endif // NDEBUG

        bool is_working = arma::all(diff < SMALLNUMBER);
        return is_working;
    }

    void time_mv() {
        arma::wall_clock timer;
        arma::vec v(HFS::Nmat, arma::fill::randu);
        timer.tic();
        arma::vec Mv(HFS::Nmat);
        HFS::MatVecProduct_func(v, Mv);
        HFS::Mv_time = timer.toc();
        std::cout << "Mv time = " << HFS::Mv_time << std::endl;
    }

    arma::mat build_matrix(double (*Matrix_func)(arma::uword, arma::uword), arma::uword N) {
        /* Given pointer to matrix function, create and return the matrix */
        arma::mat matrix(N, N);
        for (arma::uword i = 0; i < N; ++i) {
            for (arma::uword j = 0; j < N; ++j) {
                matrix(i,j) = Matrix_func(i,j);
            }
        }
        return matrix;
    }

    bool everything_works() {
        assert(mv_is_working() && "There is a problem in the matrix-vector product.");
        std::cout << "Mv is working" << std::endl;
        assert(davidson_agrees_fulldiag() && "Davidson's Algorithm Didn't get the lowest eigenvalue.");
        std::cout << "Davidson is working" << std::endl;
        return true;
    }

}
