#include <armadillo>
#include <iostream>
#include <iomanip>
#include <vector>
#include "SLEPcWrapper.h"
#include <cstdio>

arma::mat mymat;

void myarma_Matvec_Prodec(arma::vec& v, arma::vec& Mv) {
    Mv = mymat * v;
}

void test_eigs(int nmat) {
    for (int i = 0; i < nmat; ++i) {
        for (int j = 0; j < nmat; ++j) {
            if (i == j) {
                mymat(i, j) = i+1;
            } else {
                mymat(i,j) = 0.0000001;
            }
        }
    }
    arma::mat eigvecs;
    arma::vec eigvals;

    arma::eig_sym(eigvals, eigvecs, mymat);
    arma::vec sortvals = arma::sort(eigvals);

    sortvals.head(5).print("Actual evals");
}

int main(int argc, char **argv){
    arma::uword nmat;

    PetscInt N;
    std::cout << "Enter nmat:" << std::endl; std::cin >> nmat;
    mymat.set_size(nmat, nmat);
    N = nmat;

    int nguess = 10;

    std::vector< std::vector<double> > vecs(nguess, std::vector<double>(N, 0.0));
    for (int i = 0; i < nguess; ++i) {
        vecs[i][i] = 1.0;
    }
    int num_evals = 5;
    int max_subspace_size = 1000;
    int blocksize = 5;
    test_eigs(N);
    
    SLEPc::EpS myeps(argc, argv, N, myarma_Matvec_Prodec);
    myeps.SetDimensions(num_evals, max_subspace_size);
    myeps.SetTol(1E-8);
    myeps.SetBlockSize(blocksize);
    myeps.SetInitialSpace(vecs);
    myeps.Solve();
    myeps.PrintEvals();
   // myeps.PrintEvecs();
    printf("# of Values Requested: %i\n", myeps.Nevals);
    printf("# of Values Converged: %i\n", myeps.nconv);
    printf("Blocksize: %i\n", myeps.BlockSize);
    printf("Number of guesses: %i\n", myeps.nguess);
    printf("Tolerance: %8.3E\n", myeps.tol);
    myeps.clean();
    return 0;
}
