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

def generate_gc(U, X, NG, order=0):
    N = U.shape[0]
    num_vars = U.shape[1] if U.ndim > 1 else 1

    data_with_gc = np.zeros((N + 2 * NG, num_vars)) if num_vars > 1 else np.zeros(N + 2 * NG)
    X_with_gc = np.zeros(N + 2 * NG)

    data_with_gc[NG:-NG] = U
    X_with_gc[NG:-NG] = X

    if order == 0:
        for i in range(NG):
            if num_vars > 1:
                data_with_gc[i, :] = data_with_gc[NG, :]
                data_with_gc[-(i + 1), :] = data_with_gc[-NG - 1, :]
            else:
                data_with_gc[i] = data_with_gc[NG]
                data_with_gc[-(i + 1)] = data_with_gc[-NG - 1]

        dx_left = X_with_gc[NG+1] - X_with_gc[NG]
        dx_right = X_with_gc[-NG-1] - X_with_gc[-NG-2]

        for i in range(NG):
            X_with_gc[NG - 1 - i] = X_with_gc[NG - i] - dx_left # Left ghost cells
            X_with_gc[NG + N + i] = X_with_gc[NG + N + i - 1] + dx_right # Right ghost cells

    return data_with_gc, X_with_gc

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
def eigen(U):
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