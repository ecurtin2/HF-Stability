import matplotlib.pyplot as plt
import scipy.special as sp
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
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
           ,'mycase'
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
           ,'Dav_final_val'
           ,'full_diag_min'
           ,'full_diag_time'
           ,'Total Elapsed Time'
           ,'Davidson Tolerance'
           ,'Dav_minits'
           ,'Dav_maxits'
           ,'Dav_maxsubsize'
           ,'cond_number'
           ]

    vectorkeys = ['Occ Energies', 'Vir Energies', 'Excitation Energies'
                 ,'Kgrid'
                 ,'DavVals'
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

def add_Dir_to_pickle_df(pickle, dirname='log', ext='.log', moveto=None):
    """Load DataFrame from pickle, and read in files. 
    
    Optionally move files to moveto dir.
    """
    for dirpath, dirnames, files in os.walk(dirname):
        if not files:
            print "Directory is Empty, df is unchanged"
            df = pd.read_pickle(pickle)
        else:
            dataframes = []
            try:
                df1 = pd.read_pickle(pickle)
            except:
                pass
            else:
                dataframes.append(df1)

            df_fromdir = directory_to_df(dirname, ext)
            dataframes.append(df_fromdir)
            df = pd.concat(dataframes)
            df.to_pickle(pickle)
            files = glob.glob(dirname + '/*' + ext)
            if moveto:
                if os.path.isdir(moveto):
                    for f in files:
                        shutil.move(f, moveto)
                else:
                    print "Target Directory Doesn't Exist, did not move log files"
    return df

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

def analytic_exch(k, kf, ndim):
    const = -2.0 * kf / math.pi
    if ndim == 2:
        return const * f2D(k / kf)
    elif ndim == 3:
        return const * f3D(k / kf)

def analytic_energy(k, kf, ndim):
    x = np.linalg.norm(k)  #works on k of any dimension
    return (x*x / 2.0) + analytic_exch(x, kf, ndim)


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

    circle = plt.Circle((0, 0), radius=kf, fc='none', linewidth=1, zorder=4)
    sqrpoints = [[kmax, kmax]
                ,[kmax, -kmax]
                ,[-kmax, -kmax]
                ,[-kmax, kmax]]
    square =plt.Polygon(sqrpoints, edgecolor=sns.color_palette()[0], fill=None, zorder=4)
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
                c=sns.color_palette()[0], label='Occupied', s=8, zorder=3, edgecolors='none')
    vir_color = sns.color_palette()[2]
    spec_vir_color = (255.0/255, 204./255, 153./255) 

    ax.scatter(active_virs[:,0], active_virs[:,1],
                c=vir_color, label='Virtual', s=8, zorder=2, edgecolors='none')
    ax.scatter(spec_virs[:,0], spec_virs[:,1],
                c=spec_vir_color, alpha=spec_alpha, label='Spectator Virtuals', 
                s=8, zorder=1, edgecolors='none')
    ax.set_xlim(-scale*kmax, scale*kmax)
    ax.set_ylim(-scale*kmax, scale*kmax)
    rs = df.iloc[df_idx]['rs']
    Nk = df.iloc[df_idx]['Nk']
    ndim = df.iloc[df_idx]['ndim']
    title = 'rs = ' + str(rs)[:3] + ' Nk = ' + str(Nk) + ' ndim = ' + str(ndim)
    if labels:
#ax.set_title(title)
        ax.set_xlabel('k${}_x$')
        ax.set_ylabel('k${}_y$')
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
        title = 'r_s = ' + str(rs)[:3] + ' Nk = ' + str(Nk) + ' ndim = ' + str(ndim)
#       ax.set_title(title)
        ax.set_xlabel('$\epsilon_{vir} - \epsilon_{occ}$ (Hartree)')
        ax.set_ylabel('Count')
        return ax

def axplot_energy_compare(df, ax, df_idx, discretized=True, analytic=True):
    df_row = df.iloc[df_idx]
    kmax = df_row['kmax']
    ndim = df_row['ndim']
    kf = df_row.kf
    Ef = kf**2 / 2.0

    # Analytic plots
    k = np.linspace(0, kmax, 500)
    kinetic = (0.5 * k*k) / Ef
    exch    = [analytic_exch(ki, kf, ndim) for ki in k] / Ef #rescale
    energy  = [analytic_energy(ki, kf, ndim) for ki in k] / Ef
    
    k = k / kf # rescale for plot
    if analytic:
        ax.plot(k, energy , 'k-' , label='Total', zorder=1)
        ax.plot(k, kinetic, 'k:' , label='Kinetic', zorder=1)    
        ax.plot(k, exch   , 'k--', label='Exchange', zorder=1)
    
    # Discrete plots
    occ_energies = df_row['Occ Energies'] / Ef
    vir_energies = df_row['Vir Energies'] / Ef
    kgrid = df_row['Kgrid']
    occ_states = df_row['Occupied States']
    vir_states = df_row['Virtual States']
    kocc = kgrid[occ_states]
    kocc = [np.linalg.norm(ki) for ki in kocc] / kf
    kvir = kgrid[vir_states]
    kvir = [np.linalg.norm(ki) for ki in kvir] / kf
    if discretized:
        ax.scatter(kocc, occ_energies, c=sns.color_palette()[0], label='Occupied',
                   s=2, zorder=2, edgecolors='none')
        ax.scatter(kvir, vir_energies, c=sns.color_palette()[2], label='Virtual', 
                   s=2, zorder=2, edgecolors='none')
        
    # Plot Options
    ax.set_xlim(0, 2)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r'$\frac{k}{k_f}$')
    ax.set_ylabel(r'$\frac{\epsilon_k^{HF}}{\epsilon_F}$')
    return ax

def axplot_eval_convergence(df, ax, df_idx):
    df_row = df.iloc[df_idx]
    rs = df_row['rs']
    Nk = df_row['Nk']
    eigval = df_row['full_diag_min']
    ndim = df_row['ndim']
    davvals = df_row['DavVals']
    its = df_row['dav_its']
    num_evals = df_row['Dav_Num_evals']
    davvals = davvals.reshape(its, num_evals)
    its = len(davvals)
    x = range(1, its + 1)
    idx = np.argsort(davvals[:, -1])



    blues= sns.color_palette('GnBu_d', num_evals)
    reds = sns.color_palette('Reds_r', num_evals)

    cols = blues

    if abs(eigval - np.amin(davvals[-1, :])) > 1E-4:
        cols = reds

    title = 'r_s = ' + str(rs) + ' Nk = ' + str(Nk) + ' ndim = ' + str(ndim)
    ax.set_title(title)
    for i in range(num_evals):
        ax.plot(x, davvals[:, i], 'o-', c=cols[i])
    return ax

def plot_dav_vs_full(df):
    df_with_fulldiags = df[df['full_diag_min'].notnull()]
    Nks_full = df_with_fulldiags.Nk.as_matrix()
    Nks = df.Nk.as_matrix()
    davmins = df.Dav_final_val.as_matrix()
    xmin = np.amin(Nks)-1
    xmax = np.amax(Nks) + 1
    plt.xlim=(xmin, xmax)
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
