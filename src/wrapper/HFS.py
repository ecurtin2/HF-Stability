# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_HFS', [dirname(__file__)])
        except ImportError:
            import _HFS
            return _HFS
        if fp is not None:
            try:
                _mod = imp.load_module('_HFS', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _HFS = swig_import_helper()
    del swig_import_helper
else:
    import _HFS
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



def main():
    return _HFS.main()
main = _HFS.main

def main_(rs, Nk, ndim, outputfilename):
    return _HFS.main_(rs, Nk, ndim, outputfilename)
main_ = _HFS.main_

_HFS.PI_swigconstant(_HFS)
PI = _HFS.PI

_HFS.SMALLNUMBER_swigconstant(_HFS)
SMALLNUMBER = _HFS.SMALLNUMBER

def exchange(arg1, arg2):
    return _HFS.exchange(arg1, arg2)
exchange = _HFS.exchange

def two_electron(arg1, arg2):
    return _HFS.two_electron(arg1, arg2)
two_electron = _HFS.two_electron

def two_electron_check(arg1, arg2, arg3, arg4):
    return _HFS.two_electron_check(arg1, arg2, arg3, arg4)
two_electron_check = _HFS.two_electron_check

def to_first_BZ(arg1):
    return _HFS.to_first_BZ(arg1)
to_first_BZ = _HFS.to_first_BZ

def is_vir(arg1):
    return _HFS.is_vir(arg1)
is_vir = _HFS.is_vir

def k_to_index(*args):
    return _HFS.k_to_index(*args)
k_to_index = _HFS.k_to_index

def occ_idx_to_k(arg1):
    return _HFS.occ_idx_to_k(arg1)
occ_idx_to_k = _HFS.occ_idx_to_k

def vir_idx_to_k(arg1):
    return _HFS.vir_idx_to_k(arg1)
vir_idx_to_k = _HFS.vir_idx_to_k

def calc_params():
    return _HFS.calc_params()
calc_params = _HFS.calc_params

def calc_kf():
    return _HFS.calc_kf()
calc_kf = _HFS.calc_kf

def calc_vol_and_two_e_const():
    return _HFS.calc_vol_and_two_e_const()
calc_vol_and_two_e_const = _HFS.calc_vol_and_two_e_const

def calc_occ_states():
    return _HFS.calc_occ_states()
calc_occ_states = _HFS.calc_occ_states

def calc_occ_energies():
    return _HFS.calc_occ_energies()
calc_occ_energies = _HFS.calc_occ_energies

def calc_vir_energies():
    return _HFS.calc_vir_energies()
calc_vir_energies = _HFS.calc_vir_energies

def calc_energies(arg1, arg2):
    return _HFS.calc_energies(arg1, arg2)
calc_energies = _HFS.calc_energies

def calc_excitations():
    return _HFS.calc_excitations()
calc_excitations = _HFS.calc_excitations

def calc_exc_energy():
    return _HFS.calc_exc_energy()
calc_exc_energy = _HFS.calc_exc_energy

def calc_vir_N_to_1_mat():
    return _HFS.calc_vir_N_to_1_mat()
calc_vir_N_to_1_mat = _HFS.calc_vir_N_to_1_mat

def calc_inv_exc_mat():
    return _HFS.calc_inv_exc_mat()
calc_inv_exc_mat = _HFS.calc_inv_exc_mat

def matvec_prod_3A(arg1):
    return _HFS.matvec_prod_3A(arg1)
matvec_prod_3A = _HFS.matvec_prod_3A

def matvec_prod_3B(arg1):
    return _HFS.matvec_prod_3B(arg1)
matvec_prod_3B = _HFS.matvec_prod_3B

def matvec_prod_3H(arg1):
    return _HFS.matvec_prod_3H(arg1)
matvec_prod_3H = _HFS.matvec_prod_3H

def calc_1B(arg1, arg2):
    return _HFS.calc_1B(arg1, arg2)
calc_1B = _HFS.calc_1B

def calc_3B(arg1, arg2):
    return _HFS.calc_3B(arg1, arg2)
calc_3B = _HFS.calc_3B

def calc_1A(arg1, arg2):
    return _HFS.calc_1A(arg1, arg2)
calc_1A = _HFS.calc_1A

def calc_3A(arg1, arg2):
    return _HFS.calc_3A(arg1, arg2)
calc_3A = _HFS.calc_3A

def calc_3H(arg1, arg2):
    return _HFS.calc_3H(arg1, arg2)
calc_3H = _HFS.calc_3H

def kb_j_to_t(arg1, arg2):
    return _HFS.kb_j_to_t(arg1, arg2)
kb_j_to_t = _HFS.kb_j_to_t

def build_guess_evecs(N, which=0):
    return _HFS.build_guess_evecs(N, which)
build_guess_evecs = _HFS.build_guess_evecs

def davidson_wrapper(N, guess_evecs, block_size=1, which=0, num_of_roots=1, max_its=20, max_sub_size=1000, tolerance=10E-8):
    return _HFS.davidson_wrapper(N, guess_evecs, block_size, which, num_of_roots, max_its, max_sub_size, tolerance)
davidson_wrapper = _HFS.davidson_wrapper

def davidson_algorithm(arg1, arg2, arg3, arg4, arg5, arg6, arg7, matrix, matvec_product):
    return _HFS.davidson_algorithm(arg1, arg2, arg3, arg4, arg5, arg6, arg7, matrix, matvec_product)
davidson_algorithm = _HFS.davidson_algorithm

def davidson_agrees_fulldiag():
    return _HFS.davidson_agrees_fulldiag()
davidson_agrees_fulldiag = _HFS.davidson_agrees_fulldiag

def mv_is_working():
    return _HFS.mv_is_working()
mv_is_working = _HFS.mv_is_working

def everything_works():
    return _HFS.everything_works()
everything_works = _HFS.everything_works

def build_matrix():
    return _HFS.build_matrix()
build_matrix = _HFS.build_matrix

def write_output(detail=False):
    return _HFS.write_output(detail)
write_output = _HFS.write_output

def centerstring(s, width):
    return _HFS.centerstring(s, width)
centerstring = _HFS.centerstring
# This file is compatible with both classic and new-style classes.

cvar = _HFS.cvar

