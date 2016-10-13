# This File was automatically generated by CppClassWrapper
########################################################################
#                             .pyx Header                              #
########################################################################
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

from libcpp cimport bool
from libcpp.string cimport string
from libc.math cimport sqrt
import itertools
import math
from scipy import special as sp
import numpy as np
cimport numpy as np
cimport cython
import matplotlib.pyplot as plt
import seaborn as sns

# This group imports from ./lib
include "pyfiles/lib/cyarma.pyx"    
from pyfiles.lib import general_methods as gm


########################################################################
#                      Generated cpp->pyx Header                       #
########################################################################
cdef extern from "cppfiles/HFSnamespace.h" namespace "HFS":

    #Attributes
     double  bzone_length, vol, rs, kf, kmax, fermi_energy
     double  two_e_const, deltaK
     long long unsigned int Nocc, Nvir, Nexc, N_elec, Nk
     int ndim
     vec  occ_energies, vir_energies, exc_energies, kgrid
     vec inp_test_vec, out_vec1, out_vec2
     mat mattest
     umat occ_states, vir_states, excitations
     vec dav_vals
     mat dav_vecs
     mat states
     int dav_its
     string dav_message
    #Methods
     void calc_kf()
     void calc_vol_and_two_e_const()
     void print_params()
     void calc_occ_states()
     void calc_occ_energies()
     void calc_vir_energies()
     void calc_excitations()
     bool is_vir(double)
     void   calc_exc_energy()
     void   calc_energies(umat&, vec&)
     double exchange(umat&, long long unsigned int)
     double two_electron(vec, vec)
     double two_electron_check(vec, vec, vec, vec)
     double calc_1B(long long unsigned int, long long unsigned int)
     double calc_3B(long long unsigned int, long long unsigned int)
     double calc_1A(long long unsigned int, long long unsigned int)
     double calc_3A(long long unsigned int, long long unsigned int)
     double calc_3H(long long unsigned int, long long unsigned int)
     void to_first_BZ(vec&)
     void calc_params()
     void calc_inv_exc_map()
     void calc_vir_N_to_1_map()
     uvec k_to_index(vec)
     umat k_to_index(mat)
     uvec inv_exc_map_test
     void build_mattest()
     void matvec_prod_me()
     vec matvec_prod_3H(vec)
     void davidson_wrapper(long long unsigned int, long long unsigned int, long long unsigned int, long long unsigned int, mat, double, int)
     long long unsigned int kb_j_to_t(vec, long long unsigned int)
     vec matvec_prod_3A(vec)
     vec matvec_prod_3B(vec)
     vec occ_idx_to_k(long long unsigned int)
     vec vir_idx_to_k(long long unsigned int)
     void davidson_algorithm(long long unsigned int,long long unsigned int, long long unsigned int, long long unsigned int, long long unsigned int, mat, double, double (*matrix)(long long unsigned int, long long unsigned int), vec (*matvec_product)(vec v))


########################################################################
#                              pyx class                               #
########################################################################


#################################################################################
#                                                                               #
#                          Python Functions                                   # 
#                                                                               #
#################################################################################
def py_f2D(y):
    if y <= 1.0:
        #scipy and guiliani/vignale define K and E differently, x -> x*x
        return sp.ellipe(y*y)
    else:
        #scipy and guiliani/vignale define K and E differently, x -> x*x
        x = 1.0 / y
        return y * (sp.ellipe(x*x) - (1.0 - x*x) * sp.ellipk(x*x))

def py_f3D(y):
    if y < 10e-10:
        return 1.0
    return 0.5 + (1 - y*y) / (4*y) * math.log(abs((1+y) / (1-y)))

def py_analytic_exch(k):
    const = -2.0 * get_kf() / math.pi
    if get_ndim() == 2:
        return const * py_f2D(k / get_kf())
    elif get_ndim() == 3:
        return const * py_f3D(k / get_kf())

def py_analytic_energy(k):
    x = np.linalg.norm(k)  #works on k of any dimension
    return (x*x / 2.0) + py_analytic_exch(x)


#################################################################################
#                                                                               #
#                          Plotting Functions                                   # 
#                                                                               #
#################################################################################
def plot_1stBZ(spec_alpha=0.20):
    # Draw Shapes
    assert (get_ndim() == 2), 'Only 2d is supported right now'
    circle = plt.Circle((0, 0), radius=get_kf(), fc='none', linewidth=1)
    sqrpoints = [[get_kmax(), get_kmax()]
                ,[get_kmax(), -get_kmax()]
                ,[-get_kmax(), -get_kmax()]
                ,[-get_kmax(), get_kmax()]]
    square =plt.Polygon(sqrpoints, edgecolor=sns.color_palette()[0], fill=None)
    plt.gca().add_patch(circle)
    plt.gca().add_patch(square)

    # Get 'spectator virtuals'
    kvir_y = get_kgrid()[get_vir_states()[:, 1]]
    is_spec = np.logical_or((kvir_y > get_kf()), (kvir_y < -get_kf()))
    mask = np.where(is_spec)
    spec_virs = get_kgrid()[get_vir_states()[mask]]
    mask2 = np.where(np.logical_not(is_spec))
    active_virs = get_kgrid()[get_vir_states()[mask2]]

    plt.scatter(get_kgrid()[get_occ_states()[:,0]], get_kgrid()[get_occ_states()[:,1]], 
                c=sns.color_palette()[0], label='Occupied')
    plt.scatter(active_virs[:,0], active_virs[:,1],
                c=sns.color_palette()[2], label='Virtual')
    plt.scatter(spec_virs[:,0], spec_virs[:,1], 
                c=sns.color_palette()[2], alpha=spec_alpha, label='Spectator Virtuals')
    scale = 1.05
    plt.xlim(-scale*get_kmax(), scale*get_kmax())
    plt.ylim(-scale*get_kmax(), scale*get_kmax())
    plt.legend(loc='center left', bbox_to_anchor=[0.95,0.5])
    plt.axis('off')
    plt.title('The First Brillouin Zone')

def plot_energy(analytic=True, Discretized=True):
    scale = 1.2
    #Analytic plot
    xmax = 2.0 * get_kf()
    x = np.linspace(0, xmax, 500)
    energy_x = np.array([py_analytic_energy(i) for i in x]) / get_fermi_energy()
    kinetic_x = np.array([0.5 * i**2 for i in x]) / get_fermi_energy()
    exch_x = np.array([py_analytic_exch(i) for i in x]) / get_fermi_energy()
    x = x / get_kf()  #rescale for plot
    if analytic:
        plt.plot(x, energy_x, 'k-' , label='Total')
        plt.plot(x, kinetic_x, 'k:', label='Kinetic')
        plt.plot(x, exch_x, 'k--', label='Exchange')
    plt.title('Orbital Energies\n'+str(get_ndim()) + 'D, rs = ' + str(get_rs()))
    plt.xlabel(r'$\frac{k}{k_f}$')
    plt.ylabel(r'$\frac{\epsilon_k^{HF}}{\epsilon_F}$')
    plt.xlim(0, 2)
    plt.ylim(scale * np.amin(energy_x), scale * np.amax(energy_x))

    #Discretized Plot
    y = get_occ_energies() / get_fermi_energy()
    x = gm.row_norm(get_kgrid()[get_occ_states()]) / get_kf()
    if Discretized:
        plt.plot(x, y, '.', c=sns.color_palette()[0], label='Occupied')
    y = get_vir_energies() / get_fermi_energy()
    x = gm.row_norm(get_kgrid()[get_vir_states()]) / get_kf()
    if Discretized:
        plt.plot(x, y, '.', c=sns.color_palette()[2], label='Virtual')

def plot_exc_hist():
    plt.hist(get_exc_energies(), get_Nexc()/30)
    plt.title('Excitation Energy Histogram')
    plt.xlabel('$\epsilon_{vir} - \epsilon_{occ}$ (Hartree)')
    plt.ylabel('Count')


########################################################################
#                               Py Funcs                               #
########################################################################
def py_calc_kf():
    calc_kf()

def py_calc_vol_and_two_e_const():
    calc_vol_and_two_e_const()

def py_print_params():
    print_params()

def py_calc_occ_states():
    calc_occ_states()

def py_calc_occ_energies():
    calc_occ_energies()

def py_calc_vir_energies():
    calc_vir_energies()

def py_calc_excitations():
    calc_excitations()

def py_is_vir(float val):
    return is_vir(val)

def py_calc_exc_energy():
    calc_exc_energy()

def py_calc_energies(np.ndarray[long long unsigned int, ndim=2, mode="fortran"] val1, 
                     np.ndarray[double, ndim=1] val2):
    VAL1 = numpy_to_umat_d(val1)
    VAL2 = numpy_to_vec_d(val2)
    calc_energies(VAL1, VAL2)

def py_exchange(np.ndarray[long long unsigned int, ndim=2, mode="fortran"] val1, 
                int val2):
    VAL1 = numpy_to_umat_d(val1)
    return exchange(VAL1, val2)

def py_two_electron(np.ndarray[double, ndim=1] val1, 
                    np.ndarray[double, ndim=1] val2):
    VAL1 = numpy_to_vec_d(val1)
    VAL2 = numpy_to_vec_d(val2)
    return two_electron(VAL1, VAL2)

def py_two_electron_check(np.ndarray[double, ndim=1] val1, 
                          np.ndarray[double, ndim=1] val2, 
                          np.ndarray[double, ndim=1] val3, 
                          np.ndarray[double, ndim=1] val4):
    VAL1 = numpy_to_vec_d(val1)
    VAL2 = numpy_to_vec_d(val2)
    VAL3 = numpy_to_vec_d(val3)
    VAL4 = numpy_to_vec_d(val4)
    return two_electron_check(VAL1, VAL2, VAL3, VAL4)

def py_calc_1B(int val1, 
               int val2):
    return calc_1B(val1, val2)

def py_calc_3B(int val1, 
               int val2):
    return calc_3B(val1, val2)

def py_calc_1A(int val1, 
               int val2):
    return calc_1A(val1, val2)

def py_calc_3A(int val1, 
               int val2):
    return calc_3A(val1, val2)

def py_calc_3H(int val1, 
               int val2):
    return calc_3H(val1, val2)

def py_to_first_BZ(np.ndarray[double, ndim=1] val1):
    VAL1 = numpy_to_vec_d(val1)
    to_first_BZ(VAL1)

def py_calc_params():
    calc_params()

def py_calc_inv_exc_map():
    calc_inv_exc_map()

def py_calc_vir_N_to_1_map():
    calc_vir_N_to_1_map()

def py_k_to_index(np.ndarray[double, ndim=1] val1):
    VAL1 = numpy_to_vec_d(val1)
    return uvec_to_numpy(k_to_index(VAL1))

def py_k_to_index(np.ndarray[double, ndim=2, mode="fortran"] val1):
    VAL1 = numpy_to_mat_d(val1)
    return umat_to_numpy(k_to_index(VAL1))

def py_build_mattest():
    build_mattest()

def py_matvec_prod_me():
    matvec_prod_me()

def py_matvec_prod_3H(np.ndarray[double, ndim=1] val1):
    VAL1 = numpy_to_vec_d(val1)
    return vec_to_numpy(matvec_prod_3H(VAL1))

def py_davidson_wrapper(int val1, 
                        int val2, 
                        int val3, 
                        int val4, 
                        np.ndarray[double, ndim=2, mode="fortran"] val5, 
                        float val6, 
                        int val7):
    VAL5 = numpy_to_mat_d(val5)
    davidson_wrapper(val1, val2, val3, val4, VAL5, val6, val7)

def py_kb_j_to_t(np.ndarray[double, ndim=1] val1, 
                 int val2):
    VAL1 = numpy_to_vec_d(val1)
    return kb_j_to_t(VAL1, val2)

def py_matvec_prod_3A(np.ndarray[double, ndim=1] val1):
    VAL1 = numpy_to_vec_d(val1)
    return vec_to_numpy(matvec_prod_3A(VAL1))

def py_matvec_prod_3B(np.ndarray[double, ndim=1] val1):
    VAL1 = numpy_to_vec_d(val1)
    return vec_to_numpy(matvec_prod_3B(VAL1))

def py_occ_idx_to_k(int val):
    return vec_to_numpy(occ_idx_to_k(val))

def py_vir_idx_to_k(int val):
    return vec_to_numpy(vir_idx_to_k(val))



########################################################################
#                         Generated Properties                         #
########################################################################
def get_bzone_length():
    """(float) Get bzone_length"""
    global bzone_length
    return bzone_length
def set_bzone_length(value):
    """(o) Set bzone_length"""
    global bzone_length
    bzone_length = float(value)


def get_vol():
    """(float) Get vol"""
    global vol
    return vol
def set_vol(value):
    """(o) Set vol"""
    global vol
    vol = float(value)


def get_rs():
    """(float) Get rs"""
    global rs
    return rs
def set_rs(value):
    """(o) Set rs"""
    global rs
    rs = float(value)


def get_kf():
    """(float) Get kf"""
    global kf
    return kf
def set_kf(value):
    """(o) Set kf"""
    global kf
    kf = float(value)


def get_kmax():
    """(float) Get kmax"""
    global kmax
    return kmax
def set_kmax(value):
    """(o) Set kmax"""
    global kmax
    kmax = float(value)


def get_fermi_energy():
    """(float) Get fermi_energy"""
    global fermi_energy
    return fermi_energy
def set_fermi_energy(value):
    """(o) Set fermi_energy"""
    global fermi_energy
    fermi_energy = float(value)


def get_two_e_const():
    """(float) Get two_e_const"""
    global two_e_const
    return two_e_const
def set_two_e_const(value):
    """(o) Set two_e_const"""
    global two_e_const
    two_e_const = float(value)


def get_deltaK():
    """(float) Get deltaK"""
    global deltaK
    return deltaK
def set_deltaK(value):
    """(o) Set deltaK"""
    global deltaK
    deltaK = float(value)


def get_Nocc():
    """(int) Get Nocc"""
    global Nocc
    return Nocc
def set_Nocc(value):
    """(t) Set Nocc"""
    global Nocc
    Nocc = int(value)


def get_Nvir():
    """(int) Get Nvir"""
    global Nvir
    return Nvir
def set_Nvir(value):
    """(t) Set Nvir"""
    global Nvir
    Nvir = int(value)


def get_Nexc():
    """(int) Get Nexc"""
    global Nexc
    return Nexc
def set_Nexc(value):
    """(t) Set Nexc"""
    global Nexc
    Nexc = int(value)


def get_N_elec():
    """(int) Get N_elec"""
    global N_elec
    return N_elec
def set_N_elec(value):
    """(t) Set N_elec"""
    global N_elec
    N_elec = int(value)


def get_Nk():
    """(int) Get Nk"""
    global Nk
    return Nk
def set_Nk(value):
    """(t) Set Nk"""
    global Nk
    Nk = int(value)


def get_ndim():
    """(int) Get ndim"""
    global ndim
    return ndim
def set_ndim(value):
    """(t) Set ndim"""
    global ndim
    ndim = int(value)


def get_occ_energies():
    """(np.ndarray[double, ndim=1]) Get occ_energies"""
    global occ_energies
    return vec_to_numpy(occ_energies)
def set_occ_energies(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set occ_energies"""
    global occ_energies
    occ_energies = numpy_to_vec_d(value)


def get_vir_energies():
    """(np.ndarray[double, ndim=1]) Get vir_energies"""
    global vir_energies
    return vec_to_numpy(vir_energies)
def set_vir_energies(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set vir_energies"""
    global vir_energies
    vir_energies = numpy_to_vec_d(value)


def get_exc_energies():
    """(np.ndarray[double, ndim=1]) Get exc_energies"""
    global exc_energies
    return vec_to_numpy(exc_energies)
def set_exc_energies(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set exc_energies"""
    global exc_energies
    exc_energies = numpy_to_vec_d(value)


def get_kgrid():
    """(np.ndarray[double, ndim=1]) Get kgrid"""
    global kgrid
    return vec_to_numpy(kgrid)
def set_kgrid(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set kgrid"""
    global kgrid
    kgrid = numpy_to_vec_d(value)


def get_inp_test_vec():
    """(np.ndarray[double, ndim=1]) Get inp_test_vec"""
    global inp_test_vec
    return vec_to_numpy(inp_test_vec)
def set_inp_test_vec(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set inp_test_vec"""
    global inp_test_vec
    inp_test_vec = numpy_to_vec_d(value)


def get_out_vec1():
    """(np.ndarray[double, ndim=1]) Get out_vec1"""
    global out_vec1
    return vec_to_numpy(out_vec1)
def set_out_vec1(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set out_vec1"""
    global out_vec1
    out_vec1 = numpy_to_vec_d(value)


def get_out_vec2():
    """(np.ndarray[double, ndim=1]) Get out_vec2"""
    global out_vec2
    return vec_to_numpy(out_vec2)
def set_out_vec2(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set out_vec2"""
    global out_vec2
    out_vec2 = numpy_to_vec_d(value)


def get_mattest():
    """(np.ndarray[double, ndim=2, mode="fortran"]) Get mattest"""
    global mattest
    return mat_to_numpy(mattest)
def set_mattest(np.ndarray[double, ndim=2, mode="fortran"] 
                     value not None):
    """(np.ndarray[double, ndim=2, mode="fortran"]) Set mattest"""
    global mattest
    mattest = numpy_to_mat_d(value)


def get_occ_states():
    """(np.ndarray[long long unsigned int, ndim=2, mode="fortran"]) Get occ_states"""
    global occ_states
    return umat_to_numpy(occ_states)
def set_occ_states(np.ndarray[long long unsigned int, ndim=2, mode="fortran"] 
                     value not None):
    """(np.ndarray[long long unsigned int, ndim=2, mode="fortran"]) Set occ_states"""
    global occ_states
    occ_states = numpy_to_umat_d(value)


def get_vir_states():
    """(np.ndarray[long long unsigned int, ndim=2, mode="fortran"]) Get vir_states"""
    global vir_states
    return umat_to_numpy(vir_states)
def set_vir_states(np.ndarray[long long unsigned int, ndim=2, mode="fortran"] 
                     value not None):
    """(np.ndarray[long long unsigned int, ndim=2, mode="fortran"]) Set vir_states"""
    global vir_states
    vir_states = numpy_to_umat_d(value)


def get_excitations():
    """(np.ndarray[long long unsigned int, ndim=2, mode="fortran"]) Get excitations"""
    global excitations
    return umat_to_numpy(excitations)
def set_excitations(np.ndarray[long long unsigned int, ndim=2, mode="fortran"] 
                     value not None):
    """(np.ndarray[long long unsigned int, ndim=2, mode="fortran"]) Set excitations"""
    global excitations
    excitations = numpy_to_umat_d(value)


def get_dav_vals():
    """(np.ndarray[double, ndim=1]) Get dav_vals"""
    global dav_vals
    return vec_to_numpy(dav_vals)
def set_dav_vals(np.ndarray[double, ndim=1] 
                     value not None):
    """(np.ndarray[double, ndim=1]) Set dav_vals"""
    global dav_vals
    dav_vals = numpy_to_vec_d(value)


def get_dav_vecs():
    """(np.ndarray[double, ndim=2, mode="fortran"]) Get dav_vecs"""
    global dav_vecs
    return mat_to_numpy(dav_vecs)
def set_dav_vecs(np.ndarray[double, ndim=2, mode="fortran"] 
                     value not None):
    """(np.ndarray[double, ndim=2, mode="fortran"]) Set dav_vecs"""
    global dav_vecs
    dav_vecs = numpy_to_mat_d(value)


def get_states():
    """(np.ndarray[double, ndim=2, mode="fortran"]) Get states"""
    global states
    return mat_to_numpy(states)
def set_states(np.ndarray[double, ndim=2, mode="fortran"] 
                     value not None):
    """(np.ndarray[double, ndim=2, mode="fortran"]) Set states"""
    global states
    states = numpy_to_mat_d(value)


def get_dav_its():
    """(int) Get dav_its"""
    global dav_its
    return dav_its
def set_dav_its(value):
    """(t) Set dav_its"""
    global dav_its
    dav_its = int(value)


def get_dav_message():
    """(str) Get dav_message"""
    global dav_message
    return dav_message
def set_dav_message(value):
    """(r) Set dav_message"""
    global dav_message
    dav_message = str(value)


def get_inv_exc_map_test():
    """(np.ndarray[long long unsigned int, ndim=1]) Get inv_exc_map_test"""
    global inv_exc_map_test
    return uvec_to_numpy(inv_exc_map_test)
def set_inv_exc_map_test(np.ndarray[long long unsigned int, ndim=1] 
                     value not None):
    """(np.ndarray[long long unsigned int, ndim=1]) Set inv_exc_map_test"""
    global inv_exc_map_test
    inv_exc_map_test = numpy_to_uvec_d(value)




