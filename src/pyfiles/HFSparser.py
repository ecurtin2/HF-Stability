import numpy as np
import pandas as pd
import os
import glob

def str_to_float_or_int(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            raise ValueError('Argument is not string of a number!')

def file_to_df(fname, idx):
    f = open(fname, 'r')

    keys = ['Nk', 'ndim', 'rs', 'deltaK', 'kf', 'kmax', 'Nocc', 'Nvir', 'Nexc', 'dav_its', 'Dav_Lowest_Val']
    vectorkeys = ['Occ Energies', 'Vir Energies', 'Kgrid', 'All Davidson Eigenvalues at Last Iteration',
                 'Davidson lowest eigenvalues at each iteration']
    matrixkeys = ['Occupied States', 'Virtual States', 'Excitations']
    allkeys = keys + vectorkeys + matrixkeys

    dic = {}
    vectordic = {}
    vectorrange = {}
    matrixdic = {}
    for lineno, line in enumerate(f):
        for key in keys:
            if (key + " =" in line):
                str_value = line.split("=")[1]
                dic[key] = str_to_float_or_int(str_value)
        for key in vectorkeys:
            if (key in line):
                begin = lineno
                length = str_to_float_or_int(line.split(":")[1])
                vectorrange[key] = [begin, length]
        for key in matrixkeys:
            if (key in line):
                begin = lineno
                nrows, ncols = line.split(":")[1].split("x")
                nrows, ncols = str_to_float_or_int(nrows), str_to_float_or_int(ncols)
                matrixdic[key] = [begin, nrows, ncols]
    f.close()



    for key in vectorrange:
        vec = pd.read_table(fname, skiprows=vectorrange[key][0], nrows=vectorrange[key][1], squeeze=True)
        dic[key] =  vec.as_matrix() # conv from pd.series to np array

    for key in matrixdic:
        mat = pd.read_table(fname, skiprows=matrixdic[key][0],
                                  nrows=matrixdic[key][1], delim_whitespace=True)
        cleanmat = mat.dropna(axis=1, how='any')
        dic[key] = cleanmat.as_matrix()

    cols = []
    data = []

    for key in allkeys:
        cols.append(key)
        data.append(dic[key])

    df = pd.DataFrame([data], columns=cols, index=[idx])
    return df[cols]

def files_to_df(files):
    dataframes = []
    for index, f in enumerate(files):
        fname = os.path.splitext(os.path.basename(f))[0]
        dataframes.append(file_to_df(f, fname))

    return pd.concat(dataframes)

def directory_to_df(dirname='log', ext='.log'):
    files = glob.glob(dirname + '/*' + ext)
    return files_to_df(files)


if __name__ == "__main__":
    print 'Import this module and call files_to_df([filelist])'
