#include "debug.hpp"

namespace HFS{

double full_diag_min;
double Mv_time;
double full_diag_time;

bool davidsonAgreesWithFullDiag() {
    arma::wall_clock timer;
    timer.tic();
    arma::mat matrix = HFS::Matrix_generator();
    arma::vec eigvals;
    arma::mat eigvecs;
    arma::eig_sym(eigvals, eigvecs, matrix);
    HFS::full_diag_min = eigvals.min();
    HFS::full_diag_time = timer.toc();

    return true;
}

bool matrixVectorProductWorks() {
    arma::vec v(HFS::Nmat, arma::fill::randu);
    arma::vec Mv(HFS::Nmat, arma::fill::zeros);
    HFS::MatVecProduct_func(v, Mv);
    arma::mat matrix = HFS::Matrix_generator();
    arma::vec v_arma = matrix * v;
    arma::vec diff = arma::abs(Mv - v_arma);

    bool is_working = arma::all(diff < SMALLNUMBER);
    /*if (!is_working) {
        arma::uvec where = arma::find(diff >= SMALLNUMBER);
        where.print("where Mv is not working: ");
    }*/
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



bool everything_works() {
    assert(matrixVectorProductWorks()
                                    && "There is a problem in the matrix-vector product.");
    std::cout << "Mv is working" << std::endl;
    assert(davidsonAgreesWithFullDiag() && "Davidson's Algorithm Didn't get the lowest eigenvalue.");
    std::cout << "Davidson is working" << std::endl;
    return true;
}

}
