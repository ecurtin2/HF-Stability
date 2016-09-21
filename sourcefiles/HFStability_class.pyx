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
    #It looks like the variables default to 0, not undefined?
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
    x_exc = (self.kgrid + self.kmax)[1:]         #all potential +x excitations within 1st BZ
    all_exc = np.zeros((self.Nk - 1, self.ndim)) # -1 excludes the occ_state
    all_exc[:,0] = x_exc                         #only consider +x,  y and z are zero
    num_add = self.Nk - 1                        # max number of excitations in 1D
    my_ones = np.ones((num_add), dtype=int)
    N = self.Nocc * num_add
    occ_idx = np.zeros((N), dtype=np.uint64)
    vir = np.zeros((N, self.ndim), dtype=np.float64)
    i1 = 0
    i2 = num_add
    occ_states_k = self.kgrid[self.occ_states]
    for index, state in enumerate(occ_states_k):
        a = all_exc + state
        b = index * my_ones
        occ_idx[i1:i2] = b
        vir[i1:i2] = a
        i1 += num_add
        i2 += num_add
    vir_norms = np.sqrt((vir*vir).sum(axis=1))  #norm of each row
    idx= np.where((vir_norms > self.kf+10E-8) & (vir_norms <= self.kmax+10E-8))
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
    occ_idx = occ_idx[sorted_idx]  # the virs have been shuffled, update occ positions
    exc = np.column_stack((occ_idx, unique_map))
    exc = np.asfortranarray(exc, dtype=np.uint64)
    self.excitations = exc
    self.Nexc = len(exc)

def calc_exc_energies(self):
    self.c_HEG.calc_exc_energy()

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
    assert np.all(np.isclose(self.kgrid[idx], array)), 'Error in momentum to index xform'
    return idx

def get_inv_exc_map_2d(self):
    self.c_HEG.get_inv_exc_map_2d()
    test = self.inv_exc_map_2d_test
    assert np.all(test == np.arange(len(test))), 'Inverse excitation map (2D) Incorrect'

