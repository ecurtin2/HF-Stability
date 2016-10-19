import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import glob
import os


#################################################################################
#                                                                               #
#                      Parsing -> DataFrame Functions                           # 
#                                                                               #
#################################################################################

def str_to_float_or_int(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def file_to_df(fname, idx):
    f = open(fname, 'r')

    keys = ['Nk', 'ndim', 'rs', 'deltaK', 'kf', 'kmax', 
            'Nocc', 'Nvir', 'Nexc', 'dav_its', 'Dav_Final_Val', 
            'Davidson_Stopping_Criteria', 'full_diag_min', 'Total Elapsed Time', 
            'num_guess_evecs','Dav_blocksize','Dav_Num_evals' ]
    vectorkeys = ['Occ Energies', 'Vir Energies', 'Excitation Energies',
                  'Kgrid', 'All Davidson Eigenvalues at Last Iteration',
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
                                  nrows=matrixdic[key][1], delim_whitespace=False)
        cleanmat = mat.dropna(axis=1, how='any')
        cleanmat = cleanmat.as_matrix()
        cleanmat = np.asarray([[int(j) for j in i[0].split()] for i in cleanmat])
        dic[key] = cleanmat

        

    cols = []
    data = []

    for key in allkeys:
        try:
            data.append(dic[key])
        except:
            data.append(np.nan)  # pandas likes NaN for missing data
        cols.append(key)

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


#################################################################################
#                                                                               #
#                         Analytic Energy Functions                             # 
#                                                                               #
#################################################################################

def f2D(y):
    if y <= 1.0:
        #scipy and guiliani/vignale define K and E differently, x -> x*x
        return sp.ellipe(y*y)
    else:
        #scipy and guiliani/vignale define K and E differently, x -> x*x
        x = 1.0 / y
        return y * (sp.ellipe(x*x) - (1.0 - x*x) * sp.ellipk(x*x))

def f3D(y):
    if y < 10e-10:
        return 1.0
    return 0.5 + (1 - y*y) / (4*y) * math.log(abs((1+y) / (1-y)))

def analytic_exch(k):
    const = -2.0 * get_kf() / math.pi
    if get_ndim() == 2:
        return const * py_f2D(k / get_kf())
    elif get_ndim() == 3:
        return const * py_f3D(k / get_kf())

def analytic_energy(k):
    x = np.linalg.norm(k)  #works on k of any dimension
    return (x*x / 2.0) + py_analytic_exch(x)


#################################################################################
#                                                                               #
#                          Plotting Functions                                   # 
#                                                                               #
#################################################################################

def get_square_tuple(N):
    """get close to square"""
    if N >  1:
        subplt_cols = int(math.floor(np.sqrt(N)))
        subplt_rows = N / subplt_cols
        if N % subplt_cols != 0:
            subplt_rows += 1
    return subplt_rows, subplt_cols

def subplt_shaper(N, shape):
    if shape is None:
        if N == 1:
            rows = 1
            cols = 1
        else:
            rows, cols = get_square_tuple(N)
    else:
        rows, cols = shape        
    return rows, cols

def df_ApplyAxplotToRows(df, shape, axplot_func, *args, **kwargs):
    N = len(df)
    rows, cols = subplt_shaper(N, shape)    
    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(3*cols, 3*rows)
    
    if N == 1:  # make into 1x1 ndarray to fit the rest of the stuff
        axes = np.asarray([axes])  
    for idx, ax in enumerate(axes.flatten()):
        if idx == N:
            for j in range(N, rows*cols):
                fig.delaxes(axes.flatten()[j])
            break
        ax = axplot_func(df, ax, idx, *args, **kwargs)
    return fig, axes

def axplot_1stBZ(df, ax, df_idx, spec_alpha, scale, labels):
    # Draw Shapes
    kmax = df.iloc[df_idx]['kmax']
    kf   = df.iloc[df_idx]['kf']
    kgrid= df.iloc[df_idx]['Kgrid']    
    vir_states = df.iloc[df_idx]['Virtual States']    
    occ_states = df.iloc[df_idx]['Occupied States']
    
    circle = plt.Circle((0, 0), radius=kf, fc='none', linewidth=1)
    sqrpoints = [[kmax, kmax]
                ,[kmax, -kmax]
                ,[-kmax, -kmax]
                ,[-kmax, kmax]]
    square =plt.Polygon(sqrpoints, edgecolor=sns.color_palette()[0], fill=None)
    ax.add_patch(circle)
    ax.add_patch(square)

    # Get 'spectator virtuals'
    kvir_y = kgrid[vir_states[:, 1]]
    is_spec = np.logical_or((kvir_y > kf), (kvir_y < -kf))
    mask = np.where(is_spec)
    spec_virs = kgrid[vir_states[mask]]
    mask2 = np.where(np.logical_not(is_spec))
    active_virs = kgrid[vir_states[mask2]]

    ax.scatter(kgrid[occ_states[:,0]], kgrid[occ_states[:,1]], 
                c=sns.color_palette()[0], label='Occupied')
    ax.scatter(active_virs[:,0], active_virs[:,1],
                c=sns.color_palette()[2], label='Virtual')
    ax.scatter(spec_virs[:,0], spec_virs[:,1], 
                c=sns.color_palette()[2], alpha=spec_alpha, label='Spectator Virtuals')
    ax.set_xlim(-scale*kmax, scale*kmax)
    ax.set_ylim(-scale*kmax, scale*kmax)
    rs = df.iloc[df_idx]['rs']
    Nk = df.iloc[df_idx]['Nk']
    ndim = df.iloc[df_idx]['ndim']
    title = 'rs = ' + str(rs) + ' Nk = ' + str(Nk) + ' ndim = ' + str(ndim)
    if labels:
        ax.set_title(title)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.legend(loc='center left', bbox_to_anchor=[0.95, 0.5])
    sns.despine(left=True, bottom=True)
#ax.axis('off')
    return ax

def axplot_exc_hist(df, ax, df_idx, bindivisor=4):
        exc_energies = df.iloc[df_idx]['Excitation Energies']
        Nexc = df.iloc[df_idx]['Nexc']
        ax.hist(exc_energies, bins=Nexc/bindivisor)
        rs = df.iloc[df_idx]['rs']
        Nk = df.iloc[df_idx]['Nk']
        ndim = df.iloc[df_idx]['ndim']
        title = 'r_s = ' + str(rs) + ' Nk = ' + str(Nk) + ' ndim = ' + str(ndim)
        ax.set_title(title)
        ax.set_xlabel('$\epsilon_{vir} - \epsilon_{occ}$ (Hartree)')
        ax.set_ylabel('Count')
        return ax

def axplot_dav_convergence_wrt_its(df, ax, df_idx):
    davmins = df.iloc[df_idx]['Davidson lowest eigenvalues at each iteration']
    iterations = len(davmins)
    rs = df.iloc[df_idx]['rs']
    Nk = df.iloc[df_idx]['Nk']
    ndim = df.iloc[df_idx]['ndim']
    x = range(1, iterations + 1)
    title = 'rs = ' + str(rs) + ' Nk = ' + str(Nk) + ' ndim = ' + str(ndim)
    ax.scatter(x, davmins, c=sns.color_palette()[0], zorder=2)
    conv_val = np.ones(100) * davmins[-1]
    x2 = np.linspace(-1, iterations + 1, 100)
    ax.plot(x2, conv_val, 'k:', linewidth=1, zorder=1)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Lowest Eigenvalue')
    pad = 0.1
    ax.set_xlim(1 - pad, iterations + pad)
    return ax


if __name__ == "__main__":
    print 'Import this module and analyze some logfiles!'
