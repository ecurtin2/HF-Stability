import functools
import time


import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc


class PetscMatWrapper(object):

    def __init__(self, n_rows, n_cols, row_val_iterator, mpi_comm=PETSc.COMM_WORLD, *args, **kwargs):
        self.mpi_comm = mpi_comm
        self.mat = PETSc.Mat(*args, **kwargs)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_nonzeros = 0

        SLEPc
        self.mat.setSizes([n_rows, n_cols])
        self.mat.setFromOptions()
        self.mat.setUp()

        self.fill(row_val_iterator)
        self.row_val_iterator = row_val_iterator

    def fill(self, row_val_iterator):

        rstart, rend = self.mat.getOwnershipRange()
        for i in range(rstart, rend):
            t = time.time()
            for j, val in row_val_iterator(i):
                self.mat[i, j] = val

            PETSc.Sys.Print('Assigning took {} seconds.'.format(time.time() - t))
            t = time.time()
            for j, val in row_val_iterator(i):
                pass
            PETSc.Sys.Print('Just looping took {} seconds.'.format(time.time() - t))

        t_row = time.time() - t
        t = time.time()
        self.mat.assemble()
        t_assmeble = time.time() - t
        PETSc.Sys.Print('Ratio of iterator to assemble is {}'.format(t_row / t_assmeble))

    @staticmethod
    def row_generator(i_row, n_cols):
        """given a row index, return a generator for the indices and values of the row.

        This is an example of the type of data structure that fill can use to efficiently build a matrix.
        """
        def it():
            for i_col in range(0, n_cols):
                if i_row == i_col:
                    yield (i_col, i_col + 1)
                elif i_row == i_col + 1:
                    yield (i_col, 0.1)
                elif i_row == i_col - 1:
                    yield (i_col, 0.1)
        return it()

    @staticmethod
    def row_generator_to_numpy(row_generator, n_rows, n_cols):
        ary = np.zeros((n_rows, n_cols))
        for i_row in range(n_rows):
            for i_col, v in row_generator(i_row):
                ary[i_row, i_col] = v
        return ary


class SlepcEPSWrapper(object):

    def __init__(self, Mat, n_eigvals=1, n_initial=1):
        self.eps = SLEPc.EPS()
        self.Mat = Mat
        self.eps.create(self.Mat.getComm())
        self.eps.setOperators(self.Mat)
        self.n_eigvals = n_eigvals
        self.eps.setDimensions(nev=self.n_eigvals, ncv=PETSc.DECIDE, mpd=PETSc.DECIDE)
        self.eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        self.eps.setType(SLEPc.EPS.Type.JD)
        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        self.eps.max_it = 99

    def solve(self):
        self.eps.setFromOptions()
        self.eps.solve()

    def set_initial_space(self, n_initial, which='identity'):
        vecs = [self.Mat.createVecLeft() for i in range(n_initial)]

        if which == 'identity':
            for i, v in enumerate(vecs):
                v[i] = 0
                v.assemblyBegin()
                v.assemblyEnd()

        self.eps.setInitialSpace(vecs)

    def get_eigenpairs(self):
        """Return a list of all eigenpairs and errors"""
        pairs = []
        nconv = self.eps.getConverged()
        if nconv > 0:
            # Create the results vectors
            vr, wr = self.Mat.getVecs()
            vi, wi = self.Mat.getVecs()
            for i in range(nconv):
                k = self.eps.getEigenpair(i, vr, vi)
                error = self.eps.computeError(i)
                if k.imag != 0.0:
                    pairs.append((k.real, k.imag, error))
                else:
                    pairs.append((k.real, error))

        return pairs

    def __str__(self):
        s = []
        its = self.eps.getIterationNumber()
        eps_type = self.eps.getType()
        nev, ncv, mpd = self.eps.getDimensions()
        tol, maxit = self.eps.getTolerances()
        nconv = self.eps.getConverged()

        s.append("Number of iterations of the method: %d" % its)
        s.append("Solution method: %s" % eps_type)
        s.append("Number of requested eigenvalues: %d" % nev)
        s.append("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
        s.append("Number of converged eigenpairs %d" % nconv)
        s.append(str(self.get_eigenpairs()))
        return '\n'.join(s)


def main():
    n = 10

    gen = functools.partial(PetscMatWrapper.row_generator, n_cols=n)
    M = PetscMatWrapper(n, n, gen)
    E = SlepcEPSWrapper(M)

    ary = np.zeros((n, n))
    for i in range(n):
        for j, v in M.row_val_iterator(i):
            ary[i, j] = v

    np_evals = np.linalg.eigvals(ary)
    np_evals = np.sort(np_evals)
    print('Numpy eigenvals :', np_evals[:E.n_eigvals])
    print(E)

if __name__ == '__main__':
    main()