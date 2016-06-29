################################################################################
#																			   # 
#            				   C++ Land is Here								   # 
#																			   # 
################################################################################
from libcpp cimport bool
import numpy as np
cimport numpy as np
include "cyarma.pyx"
	
#C++ class
#Note only methods/attributes that want python access need to be here.
cdef extern from "stability.h" namespace "HFStability":
	cdef cppclass HEG:
		HEG() except +

		#Attrubutes
		double  bzone_length, vol, rs, kf
		long    Nocc, Nvir, Nexc, N_elec, ndim, Nk
		mat states, excitations       #arma::mat wrapped by cyarma
		umat excitations              #arma::umat not native to cyarma I added it

		#Methods
		double min_eigval(long, long, long, long, long, long, bool, double, double*)
		double two_electron(double[], double[], double[], double[])
		double two_electron_2d(double[], double[], double[], double[])

################################################################################
#																			   # 
#            				   Python Land is Here							   # 
#                         (it's all actually Cython tho)                       #
#																			   # 
################################################################################
#Python interface to c++ class
cdef class PyHEG:
	cdef HEG* c_HEG

	#Standard stuff
	def __cinit__(self):
		self.c_HEG = new HEG()
	def __dealloc__(self):
		del self.c_HEG

	#Class methods
#	def getA(self, long i, long j):
#		return self.c_HEG.get_A(i, j)
#	def min_eigval(self, 
#				   np.ndarray[double, ndim = 2] GUESS_VECS not None,
#				   long MAX_ITS = 20, 
#				   long NUM_OF_ROOTS = 1, 
#				   long MAX_SUB_SIZE = 2000, 
#				   long BLOCK_SIZE   = 1,
#				   bool DOTEST=False, 
#				   double TOLERANCE  =  10E-8):
#
#		GUESS_VECS = np.asfortranarray(GUESS_VECS)
#		
#		cdef long N, num_of_roots,  num_of_guess, max_its, max_sub_size
#		N = GUESS_VECS.shape[0]
#		num_of_roots = NUM_OF_ROOTS
#		num_of_guess = GUESS_VECS.shape[1]
#		max_its = MAX_ITS
#		max_sub_size = MAX_SUB_SIZE
#		block_size = BLOCK_SIZE
#		
#		cdef bool dotest = DOTEST	
#
#		cdef double tol = TOLERANCE
#		cdef double* guess_vecs = &GUESS_VECS[0,0]
#	
#		return self.c_HEG.min_eigval(N, max_its, max_sub_size, num_of_roots, 
#											num_of_guess, block_size, dotest, tol, guess_vecs)

	#############################################################################
	#       Define properties, allows for setting/getting them in python        #
	#############################################################################
	
	
	#Wigner-Seitz Radius
	def get_rs(self):
		return self.c_HEG.rs
	def set_rs(self, value):
		#catches typerrors
		self.c_HEG.rs = float(value)
	rs = property(get_rs, set_rs)
	
	#Fermi level
	def get_kf(self):
		return self.c_HEG.kf
	def set_kf(self, value):
		self.c_HEG.kf = float(value)
	kf = property(get_kf, set_kf)
	
	#Number of k-points per dimension
	def get_Nk(self):
		return self.c_HEG.Nk
	def set_Nk(self, value):
		self.c_HEG.Nk = int(value)
	Nk = property(get_Nk, set_Nk)

	#Number of electrons
	def get_N_elec(self):
		return self.c_HEG.N_elec
	def set_N_elec(self, value):
		self.c_HEG.N_elec = int(value)
	N_elec = property(get_N_elec, set_N_elec)

	#Number of dimensions
	def get_ndim(self):
		return self.c_HEG.ndim
	def set_ndim(self, value):
		self.c_HEG.ndim = int(value)
	ndim = property(get_ndim, set_ndim)

	#Number of occupied states
	def get_Nocc(self):
		return self.c_HEG.Nocc
	def set_Nocc(self, value):
		self.c_HEG.Nocc = int(value)
	Nocc = property(get_Nocc, set_Nocc)

	#Number of virtual states
	def get_Nvir(self):
		return self.c_HEG.Nvir
	def set_Nvir(self, value):
		self.c_HEG.Nvir = int(value)
	Nvir = property(get_Nvir, set_Nvir)

	#Number of excitations
	def get_Nexc(self):
		return self.c_HEG.Nexc
	def set_Nexc(self, value):
		self.c_HEG.Nexc = int(value)
	Nexc = property(get_Nexc, set_Nexc)

	#Total length of brillouin zone = 2pi/a
	def get_bzone_length(self):
		return self.c_HEG.bzone_length
	def set_bzone_length(self, value):
		self.c_HEG.bzone_length = float(value)
	bzone_length = property(get_bzone_length, set_bzone_length)

	#Direct lattice volume
	def get_vol(self):
		return self.c_HEG.vol
	def set_vol(self, double value):
		self.c_HEG.vol = float(value)
	vol = property(get_vol, set_vol)

	#states
	def get_states(self):
		ndarray = np.zeros((self.c_HEG.states.n_rows, self.c_HEG.states.n_cols))
		return mat_to_numpy(self.c_HEG.states, ndarray)
	def set_states(self, np.ndarray[double, ndim=2, mode="c"] inp_states not None):
		#cyarma
		self.c_HEG.states = numpy_to_mat_d(inp_states)
	states = property(get_states, set_states)

	#excitations
	def get_excitations(self):
		ndarray = np.zeros((self.c_HEG.excitations.n_rows, self.c_HEG.excitations.n_cols), dtype=np.uint32)
		return umat_to_numpy(self.c_HEG.excitations, ndarray)
	def set_excitations(self, np.ndarray[unsigned int, ndim=2, mode="c"] inp_excitations not None):
		#cyarma
		self.c_HEG.excitations = numpy_to_umat_d(inp_excitations)
	excitations = property(get_excitations, set_excitations)
