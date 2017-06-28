import numpy as np
import time

import parameters
from OrbitalHessian import OrbitalHessian


def main():
    params = parameters.Parameters(n_dimensions=1, n_k_points=7, delta_fxn_magnitude=0.001)
    H = OrbitalHessian(params)


    print(params)
    print(params.states)
    print(params.excitations)

    k1 = np.array([1.0, 2.0])
    k2 = np.array([2.0, 3.0])
    k3 = np.array([0.0, 0.0])
    k4 = np.array([0.0, -4.714])
    k1 = 0.1
    k2 = 0.2
    k3 = 0.3
    k4 = 0.4
    print("Two electron integral : ", params.eri.eval(k1, k2, k2, k1))

    np.set_printoptions(precision=4, linewidth=200)

    vals, vecs = np.linalg.eigh(H.A.as_numpy())
    print("A evals : \n{}".format(vals))

    vals, vecs = np.linalg.eigh(H.B.as_numpy())
    print("B evals : \n{}".format(vals))

    vals, vecs = np.linalg.eigh(H.as_numpy())
    print("H evals : \n{}".format(vals))


if __name__ == "__main__":
    t = time.time()
    main()
    t = time.time() - t
    print("PyHFS completed in {} seconds.".format(t))
