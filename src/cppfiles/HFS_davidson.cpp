#include "HFS_davidson.hpp"


/*   This is how I used to do it, keeping here just in case 01Nov2016
    HFS::build_guess_evecs(HFS::num_guess_evecs);
    HFS::davidson_wrapper(2*HFS::Nexc
                          ,HFS::guess_evecs
                          ,HFS::Dav_blocksize
                          ,0
                          ,HFS::Dav_Num_evals
                          ,HFS::Dav_minits
                          ,HFS::Dav_maxits
                          ,HFS::Dav_maxsubsize
                          ,HFS::Dav_tol);

*/



namespace HFS {
    arma::mat guess_evecs;
    std::string Davidson_Stopping_Criteria;
    arma::vec dav_lowest_vals, dav_vals;
    arma::mat dav_vecs;
    unsigned dav_its;
    unsigned num_guess_evecs;
    unsigned Dav_blocksize;
    unsigned Dav_Num_evals;
    arma::vec dav_iteration_timer;
    double Dav_time;
    double Dav_tol;
    double Dav_final_val;
    unsigned Dav_minits;
    unsigned Dav_maxits;
    unsigned Dav_maxsubsize;
    unsigned Dav_nconv;


    void mod_gram_schmidt(arma::vec& v, arma::mat& matrix){
        // orthogonalize vector to columns of matrix
        v /= arma::norm(v);
        for (arma::uword i = 0; i < matrix.n_cols; ++i) {
            v -= arma::dot(matrix.col(i).t(), v) * matrix.col(i);
        }
        v /= arma::norm(v);
    }



    void davidson_wrapper(arma::uword N
                         ,arma::mat   guess_evecs
                         ,unsigned block_size
                         ,unsigned which
                         ,unsigned num_of_roots
                         ,unsigned min_its
                         ,unsigned max_its
                         ,unsigned max_sub_size
                         ,double      tolerance
                         ){
      /*  davidson_algorithm(N, min_its, max_its, max_sub_size, num_of_roots, block_size, guess_evecs, tolerance,
                           &HFS::calc_3H,
                           &HFS::matvec_prod_3H); */
    }

    void build_guess_evecs (int N, int which) {
        if (which == 0) {
            HFS::guess_evecs = arma::eye<arma::mat>(2*HFS::Nexc, N);
        } else if (which == 1) {
            HFS::guess_evecs = arma::eye<arma::mat>(2*HFS::Nexc, N);
        }
    }

    void davidson_algorithm(arma::uword N
                           ,unsigned min_its
                           ,unsigned max_its
                           ,unsigned max_sub_size
                           ,unsigned num_of_roots
                           ,unsigned block_size
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

        HFS::dav_vals.set_size(max_its, num_of_roots);      // make as big as possible then resize @ end.
        dav_iteration_timer.set_size(max_its);
        dav_lowest_vals.set_size(max_its);

        unsigned nconv = 0;  // number of converged eigenpairs


        //      #define DEFLATE
        #define OLD

        arma::vec diagonals(N);
        arma::vec ones(N, arma::fill::ones);

        // Need diagonals in the diagonal preconditioning step
        for (arma::uword i = 0; i < N; ++i) {
            diagonals(i) = matrix(i,i);
        }

        arma::wall_clock timer, iteration_timer;
        timer.tic();

        //Iterate the block Davidson algorithm.
        for (arma::uword i = 0; i < max_its; ++i){
            iteration_timer.tic();

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
            arma::mat Qsub, Rsub;
            arma::qr(Qsub, Rsub, sub_evecs);
            sub_evecs = Qsub;

            //sort eigenvals, vecs by value
            arma::uvec indices = arma::sort_index(sub_evals);
            sub_evecs = sub_evecs.cols(indices);
            sub_evals = sub_evals.elem(indices);

            ritz_vecs = guess_evecs * sub_evecs;
            /*Current implementation is the Diagonally Preconditioned Residue (DPR) method.
              This is the original method proposed by Davidson
              Apparently, sometimes people simply call this the Davidson method. */

            //Get the res and append to subspace if needed.
            //Sub_size changes within this loop
            old_sub_size = sub_size;
            arma::vec norms(num_of_roots, arma::fill::zeros);

            for (arma::uword j = nconv; j < block_size; ++j){
                arma::vec ritz_vec = ritz_vecs.col(j);
                double rayq = arma::dot(ritz_vec.t(), matvec_product(ritz_vec)); // rayleigh quotient

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
                        HFS::mod_gram_schmidt(corr, guess_evecs);
                        guess_evecs = arma::join_rows(guess_evecs, corr);
                        sub_size += 1;
                    }
                }
            }


            //guess_evecs.print("guess");
            //Make sure the new guess space is orthonormal
            //arma::mat Q, R;
            //arma::qr_econ(Q, R, guess_evecs);
            //Enforce sums of elements of eigenvector matrix are positive
            //with no loss of generality.
            //for (arma::uword i = 0; i < Q.n_cols; ++i) {
            //    if (Q(i,i) < 0.0) {
            //        Q.col(i) = -1.0 * Q.col(i);
            //    }
            //}
            //Q.print("Q");
            //guess_evecs = Q;


            HFS::dav_vecs =  guess_evecs;
            for (arma::uword index = 0; index < num_of_roots; ++index){
                HFS::dav_vals(i, index) = sub_evals(index);
            }

            HFS::dav_its  = i;
            HFS::dav_lowest_vals(i) = sub_evals.min();
            HFS::dav_iteration_timer(i) = iteration_timer.toc();


            if (old_sub_size == sub_size) {
                HFS::Davidson_Stopping_Criteria = "All Norms in Block Converged";
                HFS::dav_lowest_vals.resize(i);
                HFS::dav_iteration_timer.resize(i);
                HFS::dav_vals = HFS::dav_vals.head_rows(i);
                break;
            }
            else if ((sub_size + block_size) > max_sub_size) {
                HFS::Davidson_Stopping_Criteria = "Subspace Size Too Large";
                HFS::dav_lowest_vals.resize(i);
                HFS::dav_iteration_timer.resize(i);
                HFS::dav_vals = HFS::dav_vals.head_rows(i);
                break;
            } else if ((arma::all(norms < tolerance)) && (i >= min_its)) {
                HFS::Davidson_Stopping_Criteria = "All Requested Eigenvalue Norms Converged";
                HFS::dav_lowest_vals.resize(i);
                HFS::dav_iteration_timer.resize(i);
                HFS::dav_vals = HFS::dav_vals.head_rows(i);

                break;
            } else if (i == (max_its - 1)) {
                HFS::Davidson_Stopping_Criteria = "Maximum Iterations Reached";
                HFS::dav_lowest_vals.resize(i);
                HFS::dav_iteration_timer.resize(i);
                HFS::dav_vals = HFS::dav_vals.head_rows(i);
                break;
            }
        }

    HFS::Dav_time = timer.toc();
    }

}
