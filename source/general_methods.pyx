import numpy as np
cimport numpy as np
import itertools

def gm_cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    http://stackoverflow.com/questions/1208118

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        gm_cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

'''
def one_to_three(index):
    """Return 3 indices corresponding to the 1 index mapping"""
    nx = int(floor(index/(run.Nk*run.Nk)))
    ny = int(floor(index/run.Nk - run.Nk*nx))
    nz = int(index - run.Nk * run.Nk * nx - run.Nk * ny)
    return np.asarray(nx, ny, nz)

def three_to_one(array):
    """return 1 index correspond to 3idx , mapping"""
    return run.Nk*run.Nk * array[0] + run.Nk * array[1] + array[2]

def one_to_two(index):
    """Return 2 indices corresponding to the 1 index mapping"""
    nx = int(floor(index/(run.Nk)))
    ny = int(floor(index/run.Nk - nx))
    return np.asarray(nx, ny, nz)

def two_to_one(array):
    """return 1 index correspond to 2idx , mapping"""
    return int(run.Nk * array[0] + array[1])

def N_to_one(array):
    if run.ndim == 2:
        return two_to_one(array)
    elif run.ndim == 3:
        return three_to_one(array)
def one_to_N(index):
    if run.ndim == 2:
        return one_to_two(index)
    elif run.ndim == 3:
        return one_to_three(index)
'''
