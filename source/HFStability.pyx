################################################################################
#																			   # 
#            				   C++ Land is Here								   # 
#																			   # 
################################################################################
from libcpp cimport bool
import itertools
import math
from scipy import special as sp
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
		mat states                                #arma::mat wrapped by cyarma
		umat excitations                          #arma::umat not native to cyarma I added it
		uvec occ_states, vir_states

		#Methods
		double min_eigval(long, long, long, long, long, long, bool, double, double*)
		double energy(long long unsigned int)
		double two_electron_3d(double[], double[], double[])
		double two_electron_2d(double[], double[], double[])

################################################################################
#																			   # 
#            				   Python Land is Here							   # 
#                         (it's all actually Cython tho)                       #
#																			   # 
################################################################################
#Python interface to c++ class
cdef class PyHEG:
	"""This is a class docstring"""
	cdef HEG* c_HEG

	#Standard stuff
	def __cinit__(self):
		self.c_HEG = new HEG()
	def __dealloc__(self):
		del self.c_HEG

	#All constants of calculation specified by __init__
	def __init__(self, ndim=3, rs=1.0, Nk=4):
		"""This is an init docstring"""
		self.ndim = int(ndim)
		self.rs = float(rs)
		self.Nk = int(Nk)

		#Determine other parameters
		if self.ndim == 1:
			self.kf = np.pi / (4 *self.rs)
		elif self.ndim == 2:
			self.kf = 2**0.5 /self.rs
		elif self.ndim == 3:
			self.kf = (9 * np.pi / 4)**(1./3.) * (1. / self.rs) 

		kmax = 2.0 * self.kf
		kgrid = np.linspace(-kmax, kmax, self.Nk)
		#brillioun zone is from - pi/a .. pi/a
		self.bzone_length =  2.0 * kmax
		direct_length = self.bzone_length / (2.0 * np.pi)
		self.vol = direct_length**(self.ndim)

		temp = range(self.Nk)
		states = temp
		for i in range(self.ndim - 1):
			states = itertools.product(states, temp)
		states = list(states)

		#unpack into list of tuples
		if self.ndim == 3:
			states = [(item[0][0], item [0][1], item[1]) for item in states]

		if self.ndim == 1:
			states = [[kgrid[state]] for state in states]
		else:
			states = [[kgrid[i] for i in state] for state in states]

		#Separating into occ and vir by momentum
		occ_states = []
		vir_states = []
		for index, state in enumerate(states):
			k = np.linalg.norm(state)
			if k < self.kf + 10e-8:
				occ_states.append(index)
			else:
				vir_states.append(index)

		self.Nocc = len(occ_states)
		self.Nvir = len(vir_states)
		self.states = np.asarray(states)
		self.occ_states = np.asarray(occ_states, dtype=np.uint64)
		self.vir_states = np.asarray(vir_states, dtype=np.uint64)

		#RHF ONLY
		self.N_elec = 2*self.Nocc

	#Class methods

	def two_electron_3d(self, i1, i2, i3, i4):
		"""Check k-conservation and return the value of the two-electron integral.
		
		Args:
			i1 (int): index of arbitrary state 1 in the range [0, #_States]. 
			i2 (int): index of arbitrary state 2 in the range [0, #_States]. 
			i3 (int): index of arbitrary state 3 in the range [0, #_States]. 
			i4 (int): index of arbitrary state 4 in the range [0, #_States]. 
		Returns:
			(double) The value of the two electron integral. Note that the method 
				checks for momentum conservation and returns 0.0 if this 
				condition is not met. 
		Raises:
			No exceptions are raised. 
		"""

		cdef double k1[3], k2[3], k3[3], k4[3];
		cdef double mysum = 0.0;
		
		for i in range(3):
			k1[i] = self.states[i1, i]
			k2[i] = self.states[i2, i]
			k3[i] = self.states[i3, i]
			k4[i] = self.states[i4, i]
			#momentum conservation
			mysum += (k1[i] + k2[i] - (k3[i] + k4[i]))**2
		mysum = mysum**(0.5)
		#if momentum is not conserved
		if mysum > 10E-10:
			return 0.0
		#otherwise call the c++ function
		return self.c_HEG.two_electron_3d(k1, k2, k3)

	def energy(self, long long unsigned int index):
		return self.c_HEG.energy(index)


	def p_energy(self, index):

		energy = 2.0


		return energy

	def f2D(self, y):
		if y <= 1.0:
			#scipy and guiliani/vignale define K and E differently, x -> x*x
			return sp.ellipe(y*y)
		else:
			#scipy and guiliani/vignale define K and E differently, x -> x*x
			x = 1.0 / y
			return y * (sp.ellipe(x*x) - (1.0 - x*x) * sp.ellipk(x*x))

	
	def f3D(self, y):
		if y < 10e-10:
			return 1.0
		return 0.5 + (1 - y*y) / (4*y) * math.log(abs((1+y) / (1-y)))
		
	def analytic_exch(self, k):
		const = -2. * self.kf / math.pi
		if self.ndim == 2:
			return const * self.f2D(k/self.kf)
		elif self.ndim == 3:
			return const * self.f3D(k/self.kf)

	def analytic_energy(self, k):
		#works on k of any dimension
		x = np.linalg.norm(k)
		return (x*x / 2.0) + self.analytic_exch(x)

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
	
	def get_rs(self):
		"""(float) Get or set Wigner-Seitz Radius. Setting checks type."""
		return self.c_HEG.rs
	def set_rs(self, value):
		self.c_HEG.rs = float(value)
	rs = property(get_rs, set_rs)
	
	def get_kf(self):
		"""(float) Get or set Fermi wavenumber. Setting checks type."""
		return self.c_HEG.kf
	def set_kf(self, value):
		self.c_HEG.kf = float(value)
	kf = property(get_kf, set_kf)
	
	def get_Nk(self):
		"""(int) Get or set number of k-points. Setting checks type."""
		return self.c_HEG.Nk
	def set_Nk(self, value):
		self.c_HEG.Nk = int(value)
	Nk = property(get_Nk, set_Nk)

	def get_N_elec(self):
		"""(int) Get or set number of electrons. Setting checks type."""
		return self.c_HEG.N_elec
	def set_N_elec(self, value):
		self.c_HEG.N_elec = int(value)
	N_elec = property(get_N_elec, set_N_elec)

	def get_ndim(self):
		"""(int) Get or set number of dimensions. Setting checks type.
			TODO: Setting ndim automaticall defines which functions are
			to be used for the following:
				two_electron_Xd
				energy_Xd
		"""
		return self.c_HEG.ndim

	def set_ndim(self, value):
		self.c_HEG.ndim = int(value)
	ndim = property(get_ndim, set_ndim)

	def get_Nocc(self):
		"""(int) Get or set number of occupied states. Setting checks type."""
		return self.c_HEG.Nocc
	def set_Nocc(self, value):
		self.c_HEG.Nocc = int(value)
	Nocc = property(get_Nocc, set_Nocc)

	def get_Nvir(self):
		"""(int) Get or set number of virtual states. Setting checks type."""
		return self.c_HEG.Nvir
	def set_Nvir(self, value):
		self.c_HEG.Nvir = int(value)
	Nvir = property(get_Nvir, set_Nvir)

	def get_Nexc(self):
		"""(int) Get or set number of virtual states. Setting checks type."""
		return self.c_HEG.Nexc
	def set_Nexc(self, value):
		self.c_HEG.Nexc = int(value)
	Nexc = property(get_Nexc, set_Nexc)

	def get_bzone_length(self):
		"""(float) Get or set length of brillioun zone. Setting checks type.
			This is the entire length from -pi/a .. pi/a. """
		return self.c_HEG.bzone_length
	def set_bzone_length(self, value):
		self.c_HEG.bzone_length = float(value)
	bzone_length = property(get_bzone_length, set_bzone_length)

	def get_vol(self):
		"""(float) Get or set direct lattice volume. Setting checks type."""
		return self.c_HEG.vol
	def set_vol(self, double value):
		self.c_HEG.vol = float(value)
	vol = property(get_vol, set_vol)

	def get_states(self):
		"""(np.ndarray[double, ndim=2, mode="c") Get or set state indices.
			Description:
				states is an array of shape (#_states, #_dimensions).
				states[i] returns the tuple of momenta corresponding 
				to state i where each element of the tuple corresponds to a 
				dimension. 
			Setter:
				Uses the numpy_to_mat_d functionin cyarma.pyx to convert 
				from numpy array to armadillo mat<double>. Data is not 
				copied in this step (pass by reference) and boundscheck is
				not performed. Numpy arrays are c-ordered while armadillo 
				mat is fortran-ordered. 
			Getter:
				Uses the mat_to_numpy function in cyarma.pyx to convert
				from armadillo mat<double> to numpy array. This step does
				copy data and may be slow for very large arrays. 
		"""
		ndarray = np.zeros((self.c_HEG.states.n_rows, self.c_HEG.states.n_cols))
		return mat_to_numpy(self.c_HEG.states, ndarray)
	def set_states(self, np.ndarray[double, ndim=2, mode="c"] inp_states not None):
		self.c_HEG.states = numpy_to_mat_d(inp_states)
	states = property(get_states, set_states)

	def get_occ_states(self):
		"""(np.ndarray[uint64, ndim=1, mode="c") Get or set occ state indices.
			Description:
				occ_states is an array of shape (#_occ_states).
				occ_states[i] returns the an index corresponding to the
				general state which is occupied. Thus, to get momenta of
				occupied states use states[occupied_states[i]].
			Setter:
				Uses the numpy_to_mat_d functionin cyarma.pyx to convert
				from numpy array to armadillo umat. Data is not copied
				in this step (pass by reference) and boundscheck is not
				performed. Numpy arrays are c-ordered while armadillo mat
				is fortran-ordered. 
			Getter:
				Uses the mat_to_numpy function in cyarma.pyx to convert
				from armadillo umat to numpy array. This step does
				copy data and may be slow for very large arrays. 
			Gotchas:
				The armadillo matrix holds variables of type uword, defined
				in the armadillo documentation. For C++11 this is defined
				asa 64bit unsigned int on 64bit systems and 32bit unsigned
				int on 32-bit systems. This may cause problems on different
				versions and OS, make sure Armadillo is set to use 64-bit 
				words by default. 
		"""
		ndarray = np.zeros((self.c_HEG.occ_states.n_elem), dtype=np.uint64)
		return uvec_to_numpy(self.c_HEG.occ_states, ndarray)
	def set_occ_states(self, np.ndarray[long long unsigned int, ndim=1, mode="c"] inp_occ_states not None):
		self.c_HEG.occ_states = numpy_to_uvec_d(inp_occ_states)
	occ_states = property(get_occ_states, set_occ_states)

	#vir_states
	def get_vir_states(self):
		"""(np.ndarray[uint64, ndim=1, mode="c") Get/set vir state indices.
			Description:
				vir_states is an array of shape (#_vir_states).
				vir_states[i] returns the an index corresponding to the
				general state which is virupied. Thus, to get momenta of
				virupied states use states[virupied_states[i]].
			Setter:
				Uses the numpy_to_mat_d functionin cyarma.pyx to convert
				fromnumpy array to armadillo umat. Data is not copied
				in this step (pass by reference) and boundscheck is not
				performed. Numpy arrays are c-ordered while armadillo mat
				is fortran-ordered. 
			Getter:
				Uses the mat_to_numpy function in cyarma.pyx to convert
				from armadillo umat to numpy array. This step does
				copy data and may be slow for very large arrays. 
			Gotchas:
				The armadillo matrix holds variables of type uword, defined
				in the armadillo documentation. For C++11 this is defined
				as a 64bit unsigned int on 64bit systems and 32bit unsigned
				int on 32-bit systems. This may cause problems on different
				versions and OS, make sure Armadillo is set to use 64-bit 
				words by default. 
		"""
		ndarray = np.zeros((self.c_HEG.vir_states.n_elem), dtype=np.uint64)
		return uvec_to_numpy(self.c_HEG.vir_states, ndarray)
	def set_vir_states(self, np.ndarray[long long unsigned int, ndim=1, mode="c"] inp_vir_states not None):
		self.c_HEG.vir_states = numpy_to_uvec_d(inp_vir_states)
	vir_states = property(get_vir_states, set_vir_states)

	def get_excitations(self):
		"""(np.ndarray[uint64, ndim=2, mode="c") Get/set exc indices.
			Description:
				excitations is an array of shape (#_excitations, 2).
				excitations[i] returns the pair of indices corresponding
				to the occupied (excitations[i,0]) and 
				virtual (excitations[i,1]) states. To get the momentum of
				occupied state use states[excitations[i,0]] for virtual
				state use states[excitations[i,1]]. 
			Setter:
				Uses the numpy_to_mat_d functionin cyarma.pyx to convert
				from numpy array to armadillo umat. Data is not copied
				in this step (pass by reference) and boundscheck is not
				performed. Numpy arrays are c-ordered while armadillo mat
				is fortran-ordered. 
			Getter:
				Uses the mat_to_numpy function in cyarma.pyx to convert
				from armadillo umat to numpy array. This step does
				copy data and may be slow for very large arrays. 
			Gotchas:
				The armadillo matrix holds variables of type uword, defined
				in the armadillo documentation. For C++11 this is defined
				as a 64bit unsigned int on 64bit systems and 32bit unsigned
				int on 32-bit systems. This may cause problems on different
				versions and OS, make sure Armadillo is set to use 64bit 
				words by default. 
		"""
		ndarray = np.zeros((self.c_HEG.excitations.n_rows, self.c_HEG.excitations.n_cols), dtype=np.uint64)
		return umat_to_numpy(self.c_HEG.excitations, ndarray)
	def set_excitations(self, np.ndarray[long long unsigned int, ndim=2, mode="c"] inp_excitations not None):
		self.c_HEG.excitations = numpy_to_umat_d(inp_excitations)
	excitations = property(get_excitations, set_excitations)
