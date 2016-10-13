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
