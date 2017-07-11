import argparse
import numpy as np
import sys
import time

import parameters
from OrbitalHessian import OrbitalHessian

import slepc4py
from petsc4py import PETSc
Pprint = PETSc.Sys.Print

slepc4py.init(sys.argv)


def main():
    parser = argparse.ArgumentParser(description='Python Hartree-Fock Stability of HEG.')

    parser.add_argument('-r', '--rs', type=float, help='Wigner-Seitz Radius', default=1.2)
    parser.add_argument('-n', '--nk', type=int, help='Number of k-points per dimension.', default=7)
    parser.add_argument('-d', '--ndim', type=int, help='Number of dimensions.', default=2)
    parser.add_argument('-i', '--instability', type=str, help='The type of instability.', default='cRHF2cUHF')
    parser.add_argument('--cylinder-radius', type=float, help='Radius of Cylinder in 1D', default=2)
    parser.add_argument('--delta-fxn-magnitude', type=float, help='Magnitude of delta fxn potential.', default=2)
    parser.add_argument('--method', type=str, help='Method of diagonalization', default='Numpy')

    parser.add_argument('--safe-eri', help='Ensure momentum conservation in ERI.',
                        dest='safe_eri', action='store_true')
    parser.add_argument('--no-safe-eri', help='Remove momentum conservation in ERI.',
                        dest='safe_eri', action='store_false')
    parser.set_defaults(safe_eri=True)

    # for jupyter
    parser.add_argument('-f', '--fjup', type=str)

    args = vars(parser.parse_args())

    params = parameters.Parameters(n_dimensions=args['ndim']
                                  ,n_k_points=args['nk']
                                  ,instability_type=args['instability']
                                  ,cylinder_radius=args['cylinder_radius']
                                  ,delta_fxn_magnitude=args['delta_fxn_magnitude']
                                  ,safe_eri=args['safe_eri']
                                  )
    H = OrbitalHessian(params)


    # print(params)
    # print(params.states)
    # print(params.excitations)

    np.set_printoptions(precision=4, linewidth=200)
    H.lowest_eigenvalue(method=args['method'])
    Pprint('\nMatrix Size is {}\n'.format(H.size))

    Pprint('Timings:')
    for k, v in H.timings.items():
        Pprint('{k} : {v}'.format(k=k, v=v))

    #x = H.A.fast_row_generator(1)
    #Pprint(x)

if __name__ == "__main__":
    main()
