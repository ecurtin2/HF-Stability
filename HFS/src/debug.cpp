#include "debug.hpp"

namespace HFS{

    double full_diag_min;
    double Mv_time;
    double full_diag_time;

    bool davidsonAgreesWithFullDiag() {
        arma::wall_clock timer;
        timer.tic();

        arma::mat matrix = HFS::buildMatrixFromFunction(HFS::Matrix_func, HFS::Nmat);
        arma::vec eigvals;
        arma::mat eigvecs;
        arma::eig_sym(eigvals, eigvecs, matrix);
        HFS::full_diag_min = eigvals.min();
        HFS::full_diag_time = timer.toc();
        return true;
    }

    bool matrixVectorProductWorks(void (*Mv_func)(arma::vec& v, arma::vec& Mv)
                                 , double (*M_ij)(arma::uword, arma::uword)
                                 , arma::uword N)
                                 {
        arma::vec v(N, arma::fill::randu);
        arma::vec Mv(N, arma::fill::zeros);
        Mv_func(v, Mv);
        arma::mat matrix = HFS::buildMatrixFromFunction(M_ij, N);
        arma::vec v_arma = matrix * v;
        arma::vec diff = arma::abs(Mv - v_arma);

        bool is_working = arma::all(diff < SMALLNUMBER);
        return is_working;
    }

    void timeMatrixVectorProduct() {
        arma::wall_clock timer;
        arma::vec v(HFS::Nmat, arma::fill::randu);
        timer.tic();
        arma::vec Mv(HFS::Nmat);
        HFS::MatVecProduct_func(v, Mv);
        HFS::Mv_time = timer.toc();
    }

    arma::mat buildMatrixFromFunction(double (*Matrix_func)(arma::uword, arma::uword), arma::uword N) {
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
        assert(matrixVectorProductWorks(HFS::MatVecProduct_func
                                        ,HFS::Matrix_func, HFS::Nmat)
                                        && "There is a problem in the matrix-vector product.");
        std::cout << "Mv is working" << std::endl;
        assert(davidsonAgreesWithFullDiag() && "Davidson's Algorithm Didn't get the lowest eigenvalue.");
        std::cout << "Davidson is working" << std::endl;
        return true;
    }

}
