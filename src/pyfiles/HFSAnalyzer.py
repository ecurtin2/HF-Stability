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

    keys = [
            'Computation Started'
           ,'Computation Finished'
           ,'Nk'
           ,'ndim'
           ,'rs'
           ,'deltaK'
           ,'kf'
           ,'kmax'
           ,'Nocc'
           ,'Nvir'
           ,'Nexc'
           ,'ground_state_degeneracy'
           ,'dav_its'
           ,'num_guess_evecs'
           ,'Dav_blocksize'
           ,'Dav_Num_evals'
           ,'Dav_time'
           ,'Mv_time'
           ,'Davidson_Stopping_Criteria'
           ,'Dav_Final_Val'
           ,'full_diag_min'
           ,'full_diag_time'
           ,'Total Elapsed Time'
           ,'Davidson Tolerance'
           ,'Dav_minits'
           ,'Dav_maxits'
           ,'Dav_maxsubsize'
           ,'Dav_Vals_Per_Iteration'
           ]

    vectorkeys = ['Occ Energies', 'Vir Energies', 'Excitation Energies'
                 ,'Kgrid'
                 ,'Dav Vals Per Iteration'
                 ,'All Davidson Eigenvalues at Last Iteration'
                 ,'UniqueName'
                 ,'Davidson Times Per Iteration'
                 ,'Davidson lowest eigenvalues at each iteration'
                 ]

    matrixkeys = ['Occupied States', 'Virtual States', 'Excitations']
    allkeys = keys + vectorkeys + matrixkeys

    dic = {}
    vectordic = {}
    vectorrange = {}
    matrixdic = {}
    for lineno, line in enumerate(f):
        for key in keys:
            if (key + " =" in line):
                str_value = line.split("=", 1)[1].rstrip('\n')
                dic[key] = str_to_float_or_int(str_value)
            if (key + " :" in line) or (key + ":" in line):
                str_value = line.split(": ", 1)[1].rstrip('\n')
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


def axplot_eval_convergence(df, ax, df_idx, palette='husl'):
    df_row = df.iloc[df_idx]
    rs = df_row['rs']
    Nk = df_row['Nk']
    ndim = df_row['ndim']
    davvals = df_row['UniqueName']
    its = df_row['dav_its']
    num_evals = df_row['Dav_Num_evals']
    davvals = davvals.reshape(its, num_evals)
    its = len(davvals)
    x = range(1, its + 1)
    idx = np.argsort(davvals[:, -1])
    cols = sns.color_palette(palette, num_evals)
    title = 'r_s = ' + str(rs) + ' Nk = ' + str(Nk) + ' ndim = ' + str(ndim)
    ax.set_title(title)
    for i in range(num_evals):
        ax.plot(x, davvals[:, i], 'o-', c=cols[i])
    return ax


def plot_dav_vs_full(df):
    df_with_fulldiags = df[df['full_diag_min'].notnull()]
    Nks_full = df_with_fulldiags.Nk.as_matrix()
    davmins = []
    for i in range(len(df)):
        row = df.iloc[i]
        its = row['dav_its']
        num_evals = row['Dav_Num_evals']
        davvals = row['UniqueName']
        davvals = davvals.reshape(its, num_evals)
        davmin = np.amin(davvals[-1])
        davmins.append(davmin)    

    Nks = df.Nk.as_matrix()
    xmin = np.amin(Nks)-1
    xmax = np.amax(Nks) + 1
    plt.xlim(xmin, xmax)
    fullmins= df_with_fulldiags.full_diag_min.as_matrix()
    npts = 10
    zeros = np.zeros(npts)
    x = np.linspace(xmin, xmax, npts)
    
    plt.title('Stability for rs = 1.2 in 2D')
    plt.xlabel("Number of k-points per dimension")
    plt.ylabel('Lowest Eigenvalue')
    plt.plot(x, zeros   , 'k--', zorder=1)
    plt.plot(Nks, davmins , '-o' , zorder=3, label='Davidson lowest Eigenvalue')
    plt.plot(Nks_full, fullmins, 'o' , markersize=11, c=sns.color_palette()[2], 
             zorder=2, label='Exact Lowest Eigenvalue')
    plt.legend(loc='best')

if __name__ == "__main__":
    print 'Import this module and analyze some logfiles!'
