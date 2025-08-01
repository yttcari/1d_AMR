import numpy as np

CFL = 0.5   
GAMMA = 1.4

############### Global Variable ###############
def get_gamma():
    return GAMMA

def get_CFL():
    return CFL

def update_CFL(new_CFL):
    global CFL

    CFL = new_CFL

def update_GAMMA(new_GAMMA):
    global GAMMA

    GAMMA = new_GAMMA

############### Parameter Update ###############
def prim2con(prim):
    """
    Input: primitive variable [rho, u, p]
    Output: consered variable [rho, rhou, E]
    """

    rho = prim[0]
    u = prim[1]
    p = prim[2]

    E = p / (GAMMA - 1) + 0.5 * rho * (u ** 2)

    return np.array([rho, rho * u, E])

def con2prim(con):
    """
    Input: consered variable [rho, rhou, E]
    Output: primitive variable [rho, u, p]
    """
    rho = con[0]

    # Add checks for rho being close to zero or negative
    if np.any(rho <= 0):
        print(f"Warning: Negative or zero density encountered! rho={rho}")
    
        rho = np.finfo(float).eps # smallest representable positive float

    u = con[1] / rho
    p = (con[2] - 0.5 * rho * (u ** 2)) * (GAMMA - 1)

    # Add checks for p being close to zero or negative
    if np.any(p <= 0):
        print(f"Warning: Negative or zero pressure encountered! p={p} at rho={rho}, u={u}")
        p = np.finfo(float).eps # smallest representable positive float

    return np.array([rho, u, p])

def con2prim_grid(U_values):
    # Ensure U_values is treated as a float NumPy array
    U_values = np.asarray(U_values, dtype=float)

    # Initialize prim with the correct shape and data type
    prim = np.zeros_like(U_values, dtype=float)

    for i, u in enumerate(U_values):
        rho = u[0]
        mom = u[1]
        energy = u[2]
        
        # Robustness for numerical errors
        if rho <= 0: rho = 1e-10
        u_vel = mom / rho
        internal_energy = energy - 0.5 * rho * u_vel**2
        if internal_energy < 0: 
            internal_energy = 1e-10

        pressure = internal_energy * (GAMMA - 1)
        
        # MODIFIED LINE: Assign a NumPy array directly to the row
        prim[i] = np.array([rho, u_vel, pressure], dtype=float) 
        
    return prim

def prim2con_grid(con):
    tmp = []
    for i in con:
        tmp.append(prim2con(i))

    return np.array(tmp)

# Ghost cell
def generate_gc(arr, NG, mode):
    if arr.ndim == 2:
        shape = ((NG, NG), (0, 0))
    else:
        shape = (NG, NG)

    if mode == 'outflow':
        return np.pad(arr, shape, 'edge')
    elif mode == 'periodic':
        return np.pad(arr, shape, 'wrap')        

# Estimate wavespeed
def get_wavespeed(UL):
    if len(UL.shape) == 1:
        rhoL, _, pL = con2prim(np.squeeze(UL))
    else:
        tmpL = con2prim_grid(UL)

        rhoL = tmpL[:, 0]
        pL = tmpL[:, 2]

    # Compute speed of sound
    cL = np.sqrt(GAMMA * pL / rhoL)

    return cL

# For PPM
def eigen_test(U):
    u = U[:, 1]         
    rho = U[:, 0]
    cs = get_wavespeed(U)  

    ev = np.stack([u - cs, u, u + cs], axis=-1)  

    lvec = np.stack([
        np.stack([np.zeros_like(rho), -0.5 * rho / cs, 0.5 / (cs ** 2)], axis=-1),  # u - c
        np.stack([np.ones_like(rho), np.zeros_like(rho), -1.0 / (cs ** 2)], axis=-1),  # u
        np.stack([np.zeros_like(rho),  0.5 * rho / cs, 0.5 / (cs ** 2)], axis=-1)   # u + c
    ], axis=1)

    rvec = np.stack([
        np.stack([np.ones_like(rho), -cs / rho, cs**2], axis=-1),  # u - c
        np.stack([np.ones_like(rho), np.zeros_like(rho), np.zeros_like(rho)], axis=-1),  # u
        np.stack([np.ones_like(rho), cs / rho, cs**2], axis=-1)  # u + c
    ], axis=1)

    return ev, lvec, rvec

def eigen(rho, u, p, gamma):
    """Compute the left and right eigenvectors and the eigenvalues for
    the Euler equations.

    Parameters
    ----------
    rho : ndarray
        density
    u : ndarray
        velocity
    p : ndarray
        pressure
    gamma : float
        ratio of specific heats

    Returns
    -------
    ev : ndarray
        array of eigenvalues
    lvec : ndarray
        matrix of left eigenvectors, `lvec(iwave, :)` is
        the eigenvector for wave iwave
    rvec : ndarray
        matrix of right eigenvectors, `rvec(iwave, :)` is
        the eigenvector for wave iwave
    """

    # The Jacobian matrix for the primitive variable formulation of the
    # Euler equations is
    #
    #       / u   r   0   \
    #   A = | 0   u   1/r |
    #       \ 0  rc^2 u   /
    #
    # With the rows corresponding to rho, u, and p
    #
    # The eigenvalues are u - c, u, u + c

    cs = np.sqrt(gamma * p / rho)

    ev = np.array([u - cs, u, u + cs])

    # The left eigenvectors are
    #
    #   l1 =     ( 0,  -r/(2c),  1/(2c^2) )
    #   l2 =     ( 1,     0,     -1/c^2,  )
    #   l3 =     ( 0,   r/(2c),  1/(2c^2) )
    #

    lvec = np.array([[0.0, -0.5 * rho / cs, 0.5 / cs**2],  # u - c
                     [1.0, 0.0, -1.0 / cs**2],  # u
                     [0.0, 0.5 * rho / cs, 0.5 / cs**2]])  # u + c

    # The right eigenvectors are
    #
    #       /  1  \        / 1 \        /  1  \
    # r1 =  |-c/r |   r2 = | 0 |   r3 = | c/r |
    #       \ c^2 /        \ 0 /        \ c^2 /
    #

    rvec = np.array([[1.0, -cs / rho, cs**2],  # u - c
                     [1.0, 0.0, 0.0],  # u
                     [1.0, cs / rho, cs**2]])  # u + c

    return ev, lvec, rvec
