#include "debug.hpp"

namespace HFS {

scalar full_diag_min;
scalar mv_time;
scalar full_diag_time;

bool davidsonAgreesWithFullDiag(const arma::mat& M, double& diag_time) {
        arma::wall_clock timer;
        timer.tic();
        arma::mat eigvecs;
        arma::vec eigvals;
        arma::eig_sym(eigvals, eigvecs, M);
        min_eigval = eigvals.min();
        diag_time = timer.toc();
        HFS::exact_evals = eigvals;
        return true;
}

bool matrixVectorProductWorks(const arma::mat& M, void (*Mv_func)(const arma::vec& v, arma::vec& Mv)) {

        arma::vec v(HFS::Nmat, arma::fill::randu);
        arma::vec Mv(HFS::Nmat, arma::fill::zeros);
        Mv_func(v, Mv);
        arma::vec v_arma = M * v;
        arma::vec diff = arma::abs(Mv - v_arma);
        bool works = arma::all(diff < SMALLNUMBER);
        return works;
}

double timeMatrixVectorProduct(void (*Mv_func)(const arma::vec&, arma::vec&), uint N) {
        arma::vec v(N, arma::fill::randu);
        arma::wall_clock timer;
        timer.tic();
        arma::vec Mv(N);
        Mv_func(v, Mv);
        double mv_time = timer.toc();
        return mv_time;
}
} // Namespace HFS
