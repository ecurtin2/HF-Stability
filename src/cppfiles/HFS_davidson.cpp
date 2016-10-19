#include "HFS_davidson.hpp"

namespace HFS {
    arma::mat guess_evecs;
    std::string Davidson_Stopping_Criteria;
    arma::vec dav_vals, dav_lowest_vals;
    arma::mat dav_vecs;
    int dav_its;
    int num_guess_evecs;
    int Dav_blocksize;
    int Dav_Num_evals;
    double Dav_time;



    void davidson_wrapper(arma::uword N
                         ,arma::mat   guess_evecs
                         ,arma::uword block_size
                         ,int         which
                         ,arma::uword num_of_roots
                         ,arma::uword max_its
                         ,arma::uword max_sub_size
                         ,double      tolerance
                         ){
        davidson_algorithm(N, max_its, max_sub_size, num_of_roots, block_size, guess_evecs, tolerance,
                           &HFS::calc_3H,
                           &HFS::matvec_prod_3H);
    }

    void build_guess_evecs (int N, int which) {
        if (which == 0) {
            HFS::guess_evecs = arma::eye<arma::mat>(2*HFS::Nexc, N);
        } else if (which == 1) {
            HFS::guess_evecs = arma::eye<arma::mat>(2*HFS::Nexc, N);
        }
    }

    void davidson_algorithm(arma::uword N
                           ,arma::uword max_its
                           ,arma::uword max_sub_size
                           ,arma::uword num_of_roots
                           ,arma::uword block_size
                           ,arma::mat&  guess_evecs
                           ,double      tolerance
                           ,double      (*matrix)(arma::uword, arma::uword)
                           ,arma::vec   (*matvec_product)(arma::vec& v)
                           ){

        arma::uword sub_size = guess_evecs.n_cols;
        arma::uword old_sub_size = 0;
        arma::uword num_new_vecs = sub_size;
        arma::vec old_evals;
        arma::mat old_evecs = guess_evecs;
        arma::mat Mvmat(N, 0, arma::fill::zeros);
        arma::mat ritz_vecs = guess_evecs;
        arma::mat init_guess(N, num_of_roots);
        dav_lowest_vals.set_size(max_its);

        arma::vec diagonals(N);
        arma::vec ones(N, arma::fill::ones);

        // Need diagonals in the diagonal preconditioning step
        for (arma::uword i = 0; i < N; ++i) {
            diagonals(i) = matrix(i,i);
        }

        arma::wall_clock timer;
        timer.tic();

        //Iterate the block Davidson algorithm.
        for (arma::uword i = 0; i < max_its; ++i){
            arma::mat sub_mat(sub_size, sub_size, arma::fill::zeros);
            num_new_vecs = sub_size - old_sub_size;
            for (arma::uword j = 0; j < num_new_vecs; ++j)
            {
                arma::vec guess_vec = guess_evecs.col(old_sub_size + j);
                arma::vec Matvec = matvec_product(guess_vec);
                Mvmat = arma::join_rows(Mvmat, Matvec);
            }

            sub_mat = guess_evecs.t() * Mvmat;
            //Diagonalize subspace matrix.
            arma::vec sub_evals;
            arma::mat sub_evecs;
            arma::eig_sym(sub_evals, sub_evecs, sub_mat);

            //sort eigenvals, vecs by value
            arma::uvec indices = arma::sort_index(sub_evals);
            sub_evecs = sub_evecs.cols(indices);
            sub_evals = sub_evals.elem(indices);

            //old_evecs = sub_evecs;
            ritz_vecs = guess_evecs * sub_evecs;
            /*Current implementation is the Diagonally Preconditioned Residue (DPR) method.
              As far as I know right now, this is the original method propossed by Davidson
              Apparently, sometimes people simply call this the Davidson method. */

            //Get the res and append to subspace if needed.
            //Sub_size changes within this loop
            //double sum;
            old_sub_size = sub_size;
            arma::vec norms(num_of_roots, arma::fill::zeros);
            for (arma::uword j = 0; j < block_size; ++j){
                arma::vec ritz_vec = ritz_vecs.col(j);
                double rayq = arma::dot(ritz_vec.t(), matvec_product(ritz_vec)); // x.T * M * x
                if (arma::any(arma::abs(diagonals - rayq) < SMALLNUMBER)) { // Don't try to add more
                    // if diagonal = rayq , we get a divide by zero
                } else { // continue

                    arma::vec res = matvec_product(ritz_vec) - (sub_evals(j) * ritz_vec);
                    double norm = arma::norm(res);

                    if (j < num_of_roots) {
                        norms(j) = norm;
                    }

                    if (norm > tolerance) {
                        arma::vec c = 1.0 / ((rayq * ones) - diagonals);
                        arma::vec corr = c % res; // elementwise multiply
                        //arma::vec corr(N);
                        //for (arma::uword idx = 0; idx < N; ++idx){
                        //    corr(idx) = (1.0 / (rayq - diagonals(idx))) *res(idx);
                        //}
                        corr /= arma::norm(corr);
                        guess_evecs = arma::join_rows(guess_evecs, corr);
                        sub_size += 1;
                    }
                }
            }



/*
                double x_k = 0.0;
                double rayq = 0.0;
                double sum = 0.0;
                arma::vec res(N);
                //rayleigh quotient is x.T * M * x

                for (arma::uword k = 0; k < N; ++k) {
                    x_k = ritz_vecs(k,j);
                    for (arma::uword l = 0; l < N; ++l) {
                        rayq += x_k * matrix(k,l) * ritz_vecs(l,j);
                    }
                }

                for (arma::uword k = 0; k < N; ++k) {
                    sum = 0.0;
                    for (arma::uword m = 0; m < N; ++m) {
                        if (m == k){
                            sum += (matrix(k,k) - rayq) * ritz_vecs(m,j);
                        }else{
                            sum += matrix(k,m) * ritz_vecs(m,j);
                        }
                    res(k) = sum;
                    }
                }

                //Append only the res with norm > tolerance.
                double norm = arma::norm(res);
                if (j < num_of_roots) {
                    norms(j) = norm;
                }
                if (norm > tolerance) {
                    arma::vec corr(N, arma::fill::zeros);

                    //If diagonal element = eval, get singularity
                    bool apply_corr = true;
                    for (arma::uword k = 0; k < old_sub_size; ++k) {
                        if ( fabs( matrix(k,k) - rayq ) <= SMALLNUMBER) {
                            apply_corr = false;
                        }
                    }

                    //This is the DPR corr
                    for (arma::uword k=0; k < N; ++k) {
                        double val = matrix(k,k) -  rayq;
                        corr(k) = -1.0 / val * res(k);
                    }

                    //Only add if no problems encountered
                    if (apply_corr) {
                        corr /= arma::norm(corr);
                        guess_evecs = arma::join_rows(guess_evecs, corr);
                        sub_size += 1;
                    }
                    if (j == 0) {
                        std::cout << rayq << std::endl;
                        //res.print("res");
                        //corr.print("corr");
                    }

                }
            }
            */


            //Make sure the new guess space is orthonormal
            arma::mat Q, R;
            arma::qr_econ(Q, R, guess_evecs);
            //Enforce sums of elements of eigenvector matrix are positive
            //with no loss of generality.
            for (arma::uword i = 0; i < Q.n_cols; ++i) {
                if (arma::sum(Q.col(i)) < 0.0) {
                    Q.col(i) = -1.0 * Q.col(i);
                }
            }
            guess_evecs = Q;

            HFS::dav_vecs =  guess_evecs;
            HFS::dav_vals = sub_evals;
            HFS::dav_its  = i;
            HFS::dav_lowest_vals(i) = dav_vals.min();

            double t2 = timer.toc();


            if (old_sub_size == sub_size) {
                HFS::Davidson_Stopping_Criteria = "All Norms in Block Converged";
                HFS::dav_lowest_vals.resize(i);
                break;
            }
            else if ((sub_size + block_size) > max_sub_size) {
                HFS::Davidson_Stopping_Criteria = "Subspace Size Too Large";
                HFS::dav_lowest_vals.resize(i);
                break;
            } else if (arma::all(norms < tolerance)) {
                HFS::Davidson_Stopping_Criteria = "All Requested Eigenvalue Norms Converged";
                HFS::dav_lowest_vals.resize(i);
                break;
            } else if (i == (max_its - 1)) {
                HFS::Davidson_Stopping_Criteria = "Maximum Iterations Reached";
                HFS::dav_lowest_vals.resize(i);
                break;
            }
        }

    HFS::Dav_time = timer.toc();
    }

}
