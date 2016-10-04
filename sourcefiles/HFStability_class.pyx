#All constants of calculation specified by __init__
def __init__(self, ndim=3, rs=1.0, Nk=4):
    """This is an init docstring"""
    self.rs = float(rs)
    self.ndim = int(ndim)
    self.Nk = int(Nk)
    self.get_resulting_params()

def get_resulting_params(self):
    """Given rs, ndim, Nk, determine other parameters

    Description:
            Does nothing if all 3 aren't defined, which shouldn't
            ever be possible.
    Args:
            No args
    Returns:
            Nothing
    Raises:
            No exceptions
            

    """
    # It looks like the variables default to 0, not undefined
    # Because c++ defines it
    if self.rs == 0 or self.ndim == 0 or self.Nk == 0:
        return None
    try:
        self.rs
        self.ndim
        self.Nk
    except:
        return None #Don't try to calc resulting params until all defined!
    else:
        pass #Continue on if it checks out

    ##### IMPORTANT!!! THIS ORDER MATTERS FROM HERE!!!! #######
    if self.ndim == 1:
        self.kf = np.pi / (4 * self.rs)
    elif self.ndim == 2:
        self.kf = 2**0.5 / self.rs
    elif self.ndim == 3:
        self.kf = (9 * np.pi / 4)**(1./3.) * (1. / self.rs) 

    self.kmax = 2.0 * self.kf
    self.fermi_energy = 0.5 * self.kf**2
    #brillioun zone is from - pi/a .. pi/a
    self.bzone_length =  2.0 * self.kmax
    self.kgrid = np.linspace(-self.kmax, self.kmax, self.Nk)
    self.deltaK = self.kgrid[1] - self.kgrid[0]

    self.calc_occ_states()

    if self.ndim == 3:
        self.vol = self.N_elec * 4.0 / 3.0 * np.pi * (self.rs**3)
        self.two_e_const = 4.0 * np.pi / self.vol 
    elif self.ndim == 2:
        self.vol = self.N_elec * np.pi * (self.rs**2)
        self.two_e_const = 2.0 * np.pi / self.vol 
    elif self.ndim == 1:
        self.vol = self.N_elec * 2.0 * self.rs
    
    self.calc_occ_energies()
    self.calc_vir_states()
    self.calc_vir_energies()
    self.calc_exc_energies()
    self.get_inv_exc_map()
    self.get_vir_N_to_1_map()

    assert np.all(self.occ_energies < self.fermi_energy) , (
           'Not all occupied energies are below fermi energy')
    # The converse can not be said for the virtual states, since the exchange
    # interaction reduces the energies wrt the non-interacting case. Thus some virtual 
    # states lie lower in energy compared to the unperturbed fermi level.
    # the perturbed fermi level is not calculated here, as I don't need it. 
    assert np.all(self.exc_energies > 0.0), ( 
           'Not all excitation energies are positive')
    ##### IMPORTANT!!! THIS ORDER MATTERS UNTIL HERE!!!! #######

def calc_occ_states(self):
    """ayy docstring"""
    ary_list = [self.kgrid] * self.ndim
    allcombos = np.array(gm.cartesian(ary_list))
    rownorms = np.sqrt((allcombos * allcombos).sum(axis=1))
    condition_ary = rownorms <= self.kf + 10E-8
    indices = self.k_to_index(allcombos[condition_ary])
    self.occ_states = np.asfortranarray(indices, dtype=np.uint64)
    self.Nocc = len(self.occ_states)
    #RHF ONLY
    self.N_elec = 2 * self.Nocc

def calc_possible_exc(self):
    x_exc = (self.kgrid + self.kmax)[1:]           # all potential +x excitations within 1st BZ
    x_exc = np.append(x_exc, (self.kgrid - self.kmax)[1:]) # -x
    N_exc_per_occ = len(x_exc)
    all_exc = np.zeros((N_exc_per_occ, self.ndim))    # -1 excludes the occ_state
    all_exc[:,0] = x_exc                           # only consider +x,  y and z are zero
    my_ones = np.ones((N_exc_per_occ), dtype=int)
    N_exc = N_exc_per_occ * self.Nocc
    occ_idx = np.zeros((N_exc), dtype=np.uint64)
    vir = np.zeros((N_exc, self.ndim), dtype=np.float64)
    i1 = 0
    i2 = N_exc_per_occ
    occ_states_k = self.kgrid[self.occ_states]
    for index, state in enumerate(occ_states_k):
        a = all_exc + state   # add all excitations to state row-by-row
        b = index * my_ones   
        occ_idx[i1:i2] = b
        vir[i1:i2] = a
        i1 += N_exc_per_occ
        i2 += N_exc_per_occ
    vir_norms = np.sqrt((vir*vir).sum(axis=1))  #norm of each row
    in_firstBZ = np.all(((vir > (-self.kmax - 10E-10)) & (vir < (self.kmax -10E-10))), axis=1)
    is_vir = vir_norms > (self.kf + 10E-10)
    idx = np.where(is_vir & in_firstBZ)
    vir = vir[idx]          # keep only those above fermi but below cutoff
    occ_idx = occ_idx[idx]  # this is the occupied state that generated the vir
    return occ_idx, self.k_to_index(vir)

def unique_rows(self, data):                 
    """Return only unique rows
    see http://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array"""
    sorted_idx = np.lexsort(data.T)
    sorted_data =  data[sorted_idx,:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    new_idx = 0
    #EASY SPEED HERE PREALLOCATE BUT PROBS NOT NEEDED
    # unique_map maps the unfiltered row to the filtered row
    unique_map = [0]
    for i in row_mask[1:]:
        if i:
            new_idx += 1
        unique_map.append(new_idx)

    unique_map = np.asarray(unique_map, dtype=int)
    return sorted_data[row_mask], unique_map, sorted_idx

def calc_vir_states(self):
    occ_idx, non_unique_virs = self.calc_possible_exc()
    unique_virs, unique_map, sorted_idx = self.unique_rows(non_unique_virs) #keep only unique

    self.vir_states = np.asfortranarray(unique_virs, dtype=np.uint64)
    self.Nvir = len(unique_virs)
    occ_idx = occ_idx[sorted_idx]  # the virs have been shuffled, update occ positions
    exc = np.column_stack((occ_idx, unique_map))
    exc = np.asfortranarray(exc, dtype=np.uint64)
    self.excitations = exc
    self.Nexc = len(exc)

def calc_exc_energies(self):
    self.c_HEG.calc_exc_energy()
    #sort excitations in ascending order
    idx = np.argsort(self.exc_energies)
    self.excitations  = np.asfortranarray(self.excitations[idx])
    self.exc_energies = np.asfortranarray(self.exc_energies[idx])


def calc_occ_energies(self):
    self.c_HEG.calc_energy_wrap(False) # False = occupied energies

def calc_vir_energies(self):
    self.c_HEG.calc_energy_wrap(True) # True = virtual energies

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
    const = -2.0 * self.kf / math.pi
    if self.ndim == 2:
        return const * self.f2D(k/self.kf)
    elif self.ndim == 3:
        return const * self.f3D(k/self.kf)

def analytic_energy(self, k):
    x = np.linalg.norm(k)  #works on k of any dimension
    return (x*x / 2.0) + self.analytic_exch(x)

def k_to_index(self, array):
    idx = np.rint(((array + self.kmax) / self.deltaK)).astype(np.uint64)
    assert np.all(np.isclose(self.kgrid[idx], array)), 'Error in momentum to index transform.'
    return idx

def get_inv_exc_map(self):
    self.c_HEG.get_inv_exc_map()
    test = self.inv_exc_map_test
    assert np.all(test == np.arange(len(test))), 'Inverse excitation map (2D) Incorrect.'
    
def matvec_prod_arma(self):
    self.c_HEG.build_mattest()
    self.c_HEG.matvec_prod_arma()
    
def mv_is_working(self):
    vec = np.random.rand(2 * self.Nexc)
    self.inp_test_vec = vec
    self.c_HEG.build_mattest()
    self.c_HEG.matvec_prod_arma()
    self.c_HEG.matvec_prod_me()
    assert np.all(np.isclose(self.out_vec1, self.out_vec2)), 'Matrix Vector Disagrees With Arma'
    return True

def profile(self, func): 
    import pstats, cProfile
    cProfile.runctx(func, globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

def davidson(self, guess_evecs=None, which=0, tolerance=10E-8, maxits=50, maxsubsize=None, numroots=1, blocksize=1):
    """Run the davidson algorithm

    Description:
            Wrapper for the davidson algorithm
    Args:
            (int) which:
                Determines which case of stability
                    0 - (Default) Triplet H
                            
            (2D ndarray) guess_evecs
            (float)      tolerance
            (int)        maxits
            (int)        maxsubsize
            (int)        numroots
            (int)        blocksize            
    Returns:
            Nothing
    Raises:
            No exceptions
    """

    cdef long long unsigned int MAXITS, MAXSUBSIZE, NUMROOTS, BLOCKSIZE
    cdef double TOLERANCE
    cdef mat GUESS_EVECS
    cdef int WHICH
    if guess_evecs == None:
        guess_evecs = np.asfortranarray(np.eye(2 * self.Nexc, blocksize))
    if maxsubsize == None:
        maxsubsize = int(np.round(guess_evecs.shape[0] / 2))

    assert (blocksize <= guess_evecs.shape[1]), 'Must guess at least as many vectors as blocksize'
    MAXITS      = maxits
    MAXSUBSIZE  = maxsubsize
    NUMROOTS    = numroots
    BLOCKSIZE   = blocksize
    TOLERANCE   = tolerance
    GUESS_EVECS = numpy_to_mat_d(guess_evecs)
    WHICH       = which
    self.c_HEG.davidson_wrapper(MAXITS, MAXSUBSIZE, NUMROOTS, 
                                BLOCKSIZE, GUESS_EVECS, TOLERANCE, WHICH)



#################################################################################
#                                                                               #
#                          Plotting Functions                                   # 
#                                                                               #
#################################################################################

def plot_1stBZ(self, spec_alpha=0.20):
    # Draw Shapes
    assert (self.ndim == 2), 'Only 2d is supported right now'
    circle = plt.Circle((0, 0), radius=self.kf, fc='none', linewidth=1)
    sqrpoints = [[self.kmax, self.kmax]
                ,[self.kmax, -self.kmax]
                ,[-self.kmax, -self.kmax]
                ,[-self.kmax, self.kmax]]
    square =plt.Polygon(sqrpoints, edgecolor=sns.color_palette()[0], fill=None)
    plt.gca().add_patch(circle)
    plt.gca().add_patch(square)

    # Get 'spectator virtuals'
    allpts = gm.cartesian([self.kgrid[:self.Nk-1], self.kgrid[:self.Nk-1]])
    mask = []
    for idx, pt in enumerate(allpts):
        occ = np.isclose(pt, self.kgrid[self.occ_states]).all(axis=1).any()
        vir = np.isclose(pt, self.kgrid[self.vir_states]).all(axis=1).any()
        if not (occ or vir):
            mask.append(idx)
    mask = np.asarray(mask)
    spec_virs = allpts[mask]

    plt.scatter(self.kgrid[self.occ_states[:,0]], self.kgrid[self.occ_states[:,1]], 
                c=sns.color_palette()[0], label='Occupied')
    plt.scatter(self.kgrid[self.vir_states[:,0]], self.kgrid[self.vir_states[:,1]],
                c=sns.color_palette()[2], label='Virtual')
    plt.scatter(spec_virs[:,0], spec_virs[:,1], 
                c=sns.color_palette()[2], alpha=spec_alpha, label='Spectator Virtuals')
    scale = 1.05
    plt.xlim(-scale*self.kmax, scale*self.kmax)
    plt.ylim(-scale*self.kmax, scale*self.kmax)
    plt.legend(loc='center left', bbox_to_anchor=[0.95,0.5])
    plt.axis('off')
    plt.title('The First Brillouin Zone')

def plot_energy(self, analytic=True, Discretized=True):
    scale = 1.2
    #Analytic plot
    xmax = 2.0 * self.kf
    x = np.linspace(0, xmax, 500)
    energy_x = np.array([self.analytic_energy(i) for i in x]) / self.fermi_energy
    kinetic_x = np.array([0.5 * i**2 for i in x]) / self.fermi_energy
    exch_x = np.array([self.analytic_exch(i) for i in x]) / self.fermi_energy
    x = x / self.kf  #rescale for plot
    if analytic:
        plt.plot(x, energy_x, 'k-' , label='Total')
        plt.plot(x, kinetic_x, 'k:', label='Kinetic')
        plt.plot(x, exch_x, 'k--', label='Exchange')
    plt.title('Orbital Energies\n'+str(self.ndim) + 'D, rs = ' + str(self.rs))
    plt.xlabel(r'$\frac{k}{k_f}$')
    plt.ylabel(r'$\frac{\epsilon_k^{HF}}{\epsilon_F}$')
    plt.xlim(0, 2)
    plt.ylim(scale * np.amin(energy_x), scale * np.amax(energy_x))

    #Discretized Plot
    y = self.occ_energies / self.fermi_energy
    x = gm.row_norm(self.kgrid[self.occ_states]) / self.kf
    if Discretized:
        plt.plot(x, y, '.', c=sns.color_palette()[0], label='Occupied')
    y = self.vir_energies / self.fermi_energy
    x = gm.row_norm(self.kgrid[self.vir_states]) / self.kf
    if Discretized:
        plt.plot(x, y, '.', c=sns.color_palette()[2], label='Virtual')

def plot_exc_hist(self):
    plt.hist(self.exc_energies, self.Nexc/30)
    plt.title('Excitation Energy Histogram')
    plt.xlabel('$\epsilon_{vir} - \epsilon_{occ}$ (Hartree)')
    plt.ylabel('Count')

def mvprod(self, inp):
    cdef vec inpv = numpy_to_vec_d(np.asfortranarray(inp))
    cdef vec outp = self.c_HEG.matvec_prod_3H(inpv)
    return vec_to_numpy(outp)

def get_inv_exc_map(self):
    self.c_HEG.get_inv_exc_map()

def get_vir_N_to_1_map(self):
    self.c_HEG.get_vir_N_to_1_map()

