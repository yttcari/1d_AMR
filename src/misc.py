import numpy as np
import analytical

CFL = 0.4   
GAMMA = 1.4

Q_RHO = 0
Q_U = 1
Q_P = 2

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

def generate_X_gc(X, NG, bc_type):
    N = len(X)
    X_with_gc = np.zeros(N + NG * 2)

    X_with_gc[NG:-NG] = X

    dx_left = X_with_gc[NG+1] - X_with_gc[NG]
    dx_right = X_with_gc[-NG-1] - X_with_gc[-NG-2]

    for i in range(NG):
        X_with_gc[NG - 1 - i] = X_with_gc[NG - i] - dx_left # Left ghost cells
        X_with_gc[NG + N + i] = X_with_gc[NG + N + i - 1] + dx_right # Right ghost cells

    return X_with_gc    

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

def ms(compu_data, analytic_data, dx):
        numerator = np.sum(np.abs(compu_data - analytic_data) * dx)
        return (numerator / np.sum(dx)) ** 2

def calc_L1(U, X, t, dx, type, init_con):
   
    grid_solution = U
    
    rho_numeric = grid_solution[:, 0]
    u_numeric = grid_solution[:, 1]
    P_numeric = grid_solution[:, 2]
    if type == 'sod':
        rho_analytic, u_analytic, P_analytic = analytical.get_sod_solution(X, t, *init_con)
    elif type == 'plane':
        rho_analytic, u_analytic, P_analytic = analytical.get_plane_wave_solution(X, t, *init_con)
    else:
        raise ValueError("The entered problem type has not been implemented")

    rho_MSE = ms(rho_numeric, rho_analytic, dx) 
    u_MSE = ms(u_numeric, u_analytic, dx) 
    P_MSE = ms(P_numeric, P_analytic, dx) 

    return np.sqrt(rho_MSE + u_MSE + P_MSE)

def calc_L1_grid(grid, type, init_con):

    grid_cells = grid.get_all_active_cells()

    X = np.array([c.x for c in grid_cells])
    dx = np.array([c.dx for c in grid_cells])
    t = grid.t
    grid_solution = np.array([c.prim for c in grid_cells])
    
    return calc_L1(grid_solution, X, t, dx, type, init_con)

def minmod(r):
    return np.maximum(0, np.minimum(1, r))

def second_d(q, x):

    h1 = np.diff(x)[:-1]
    h2 = np.diff(x)[1:]
    
    q_i = q[1:-1]
    q_ip1 = q[2:]
    q_im1 = q[:-2]
    
    numerator = h1 * q_ip1 + h2 * q_im1 - (h1 + h2) * q_i
    denominator = h1 * h2 * (h1 + h2)
    d2q_dx2 = 2 * numerator / denominator
    return d2q_dx2
